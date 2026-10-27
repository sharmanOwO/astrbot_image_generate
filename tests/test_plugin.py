"""Run with: python -m unittest discover -s image_generate/tests -v.

AstrBot entry points are stubbed; HTTP tests use real requests against localhost.
"""
import asyncio
import base64
from email.parser import BytesParser
from email.policy import default
import importlib.util
import inspect
import json
import logging
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


def decorator(*args, **kwargs):
    def register(fn):
        fn.registration = (args, kwargs)
        return fn
    return register


class Image:
    def __init__(self, data):
        self.data = data

    async def convert_to_base64(self):
        return base64.b64encode(self.data).decode()


class Reply:
    def __init__(self, chain):
        self.chain = chain


names = ['astrbot', 'astrbot.api', 'astrbot.api.message_components',
         'astrbot.api.event', 'astrbot.api.star', 'astrbot.core',
         'astrbot.core.star', 'astrbot.core.star.filter', 'astrbot.core.star.filter.command']
modules = {name: ModuleType(name) for name in names}
modules['astrbot.api'].AstrBotConfig = dict
modules['astrbot.api'].logger = logging.getLogger('test')
modules['astrbot.api.message_components'].Image = Image
modules['astrbot.api.message_components'].Reply = Reply
modules['astrbot.api.event'].AstrMessageEvent = object
modules['astrbot.api.event'].MessageEventResult = object
modules['astrbot.api.event'].filter = SimpleNamespace(command=decorator, llm_tool=decorator)
modules['astrbot.api.star'].Star = type('Star', (), {'__init__': lambda self, context: None})
modules['astrbot.api.star'].Context = object
modules['astrbot.api.star'].register = decorator
class GreedyStr(str):
    pass


modules['astrbot.core.star.filter.command'].GreedyStr = GreedyStr
spec = importlib.util.spec_from_file_location('image_plugin_under_test', Path(__file__).parents[1] / 'main.py')
plugin_module = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', modules):
    spec.loader.exec_module(plugin_module)
Plugin = plugin_module.ImageGeneratePlugin
Error = plugin_module.ImageGenerationError


def png(width, height):
    # Header fixture: only the existing dimension reader needs to inspect it.
    return b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + width.to_bytes(4, 'big') + height.to_bytes(4, 'big')


class CommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plugin = Plugin(None, {'api_key': 'test'})
        self.images = [Image(png(1001, 777)), Image(png(300, 200))]
        self.event = SimpleNamespace(
            get_messages=lambda: [Reply([self.images[0]]), self.images[1]],
            plain_result=lambda text: text,
            get_platform_name=lambda: 'aiocqhttp',
            get_sender_id=lambda: '12345678',
        )
        self.plugin._start_edit_task = Mock()

    async def test_default_and_all_size_aliases(self):
        for text, size in [('a cat', '1024x1024')] + [
            (alias + ' a cat', size) for alias, size in plugin_module.SIZE_ALIASES.items()
        ]:
            with self.subTest(text=text):
                result = [r async for r in self.plugin.edit_image(self.event, text)]
                self.assertTrue(result)
                args = self.plugin._start_edit_task.call_args.args
                self.assertEqual(args[1], 'a cat')
                self.assertEqual(args[-1], size)
                self.assertEqual(len(args[-2]), 2)

    async def test_reference_index_and_correction(self):
        for option, expected in [('-', '1008x784'), ('-[]', '1008x784'),
                                 ('-[0]', '1008x784'), ('-[1]', '1200x800')]:
            with self.subTest(option=option):
                _ = [r async for r in self.plugin.edit_image(self.event, option + ' paint')]
                args = self.plugin._start_edit_task.call_args.args
                self.assertEqual(args[-1], expected)
                self.assertEqual(len(args[-2]), 2)

    async def test_errors_do_not_start_requests(self):
        for text in ['', '-s', '-[2] paint', '-[-1] paint', '-[abc] paint', '-z paint']:
            with self.subTest(text=text):
                self.plugin._start_edit_task.reset_mock()
                result = [r async for r in self.plugin.edit_image(self.event, text)]
                self.assertIn('失败', result[0])
                self.plugin._start_edit_task.assert_not_called()
        self.event.get_messages = lambda: []
        with patch.object(self.plugin, '_sender_avatar_upload', side_effect=Error('头像不可用')):
            _ = [r async for r in self.plugin.edit_image(self.event, 'paint')]
        self.plugin._start_edit_task.assert_not_called()

    async def test_avatar_fallback_and_sizes(self):
        # A text-only quote also counts as having no reference images.
        self.event.get_messages = lambda: [Reply([])]
        for option, expected in [('', '1024x1024'), ('-v ', '1024x1536'),
                                 ('- ', '1008x784'), ('-[] ', '1008x784'),
                                 ('-[0] ', '1008x784')]:
            with self.subTest(option=option), patch.object(
                self.plugin, '_download_image', return_value=png(1001, 777)
            ) as download:
                result = [r async for r in self.plugin.edit_image(self.event, option + 'paint')]
                self.assertIn('你的 QQ 头像', result[0])
                self.assertEqual(download.call_args.args[1], 'https://q1.qlogo.cn/g?b=qq&nk=12345678&s=640')
                args = self.plugin._start_edit_task.call_args.args
                self.assertEqual(args[-1], expected)
                self.assertEqual(args[-2], [(png(1001, 777), 'avatar.png', 'image/png')])

    async def test_explicit_images_never_download_avatar(self):
        with patch.object(self.plugin, '_sender_avatar_upload', new_callable=AsyncMock) as avatar:
            for components in [[self.images[0]], [Reply([self.images[1]])]]:
                self.event.get_messages = lambda: components
                _ = [r async for r in self.plugin.edit_image(self.event, 'paint')]
            avatar.assert_not_awaited()
            # A supplied broken image must fail instead of silently changing the subject.
            self.event.get_messages = lambda: [Image(b'invalid image')]
            self.plugin._start_edit_task.reset_mock()
            result = [r async for r in self.plugin.edit_image(self.event, 'paint')]
            self.assertIn('失败', result[0])
            avatar.assert_not_awaited()
            self.plugin._start_edit_task.assert_not_called()

    async def test_avatar_download_failures_and_invalid_content(self):
        self.event.get_messages = lambda: []
        for data in [b'', b'<html>error</html>', b'\x89PNG\r\n\x1a\n', png(0, 100)]:
            with self.subTest(data=data), patch.object(self.plugin, '_download_image', return_value=data):
                result = [r async for r in self.plugin.edit_image(self.event, 'paint')]
                self.assertIn('头像失败', result[0])
                self.plugin._start_edit_task.assert_not_called()
        with patch.object(self.plugin, '_download_image', side_effect=Error('下载超时')):
            result = [r async for r in self.plugin.edit_image(self.event, 'paint')]
            self.assertIn('头像失败', result[0])
            self.plugin._start_edit_task.assert_not_called()

    async def test_avatar_platform_id_and_index_errors(self):
        self.event.get_messages = lambda: []
        with patch.object(self.plugin, '_download_image', return_value=png(640, 640)) as download:
            self.event.get_platform_name = lambda: 'discord'
            result = [r async for r in self.plugin.edit_image(self.event, 'paint')]
            self.assertIn('当前平台', result[0])
            download.assert_not_called()
            self.event.get_platform_name = lambda: 'aiocqhttp'
            self.event.get_sender_id = lambda: ''
            result = [r async for r in self.plugin.edit_image(self.event, 'paint')]
            self.assertIn('QQ 号', result[0])
            download.assert_not_called()
            self.event.get_sender_id = lambda: '12345678'
            result = [r async for r in self.plugin.edit_image(self.event, '-[1] paint')]
            self.assertIn('只有 1 张参考图', result[0])
            self.plugin._start_edit_task.assert_not_called()

    async def test_avatar_download_runs_in_worker_and_tool_stays_unchanged(self):
        from threading import get_ident
        caller = get_ident()
        workers = []
        def download(*args):
            workers.append(get_ident())
            return png(640, 640)
        self.event.get_messages = lambda: []
        with patch.object(self.plugin, '_download_image', side_effect=download):
            _ = [r async for r in self.plugin.edit_image(self.event, 'paint')]
            self.assertTrue(workers and all(worker != caller for worker in workers))
        with patch.object(self.plugin, '_sender_avatar_upload', new_callable=AsyncMock) as avatar:
            result = [r async for r in self.plugin.edit_image_tool(self.event, 'paint')]
            self.assertIn('需要引用或包含图片', result[0])
            avatar.assert_not_awaited()

    async def test_text_command(self):
        calls = []
        async def generate(event, prompt, size):
            calls.append((prompt, size))
            yield 'started'
        self.plugin.generate_image = generate
        for option, expected in [('', '1024x1024')] + list(plugin_module.SIZE_ALIASES.items()):
            result = [r async for r in self.plugin.text_image(self.event, option + ' a cat')]
            self.assertEqual(result, ['started'])
            self.assertEqual(calls[-1], ('a cat', expected))
        result = [r async for r in self.plugin.text_image(self.event, '-[0] a cat')]
        self.assertIn('仅支持图生图', result[0])

    def test_registration(self):
        self.assertEqual(Plugin.text_image.registration, (('文生图',), {'alias': {'wst', 'w'}}))
        self.assertEqual(Plugin.edit_image.registration, (('图生图',), {'alias': {'tst', 't', 'tushengtu'}}))

    async def test_greedy_signature_preserves_command_tail(self):
        # AstrBot records the default instead of the annotation when one exists.
        # Keep the GreedyStr marker intact for BOTH command handlers.
        for handler in [Plugin.text_image, Plugin.edit_image]:
            parameter = inspect.signature(handler).parameters['prompt']
            marker = (parameter.annotation if parameter.default is inspect.Parameter.empty
                      else parameter.default)
            self.assertIs(marker, GreedyStr)
        tokens = '- 将图一角色换成图二 保留背景'.split()
        prompt = ' '.join(tokens)  # AstrBot's GreedyStr conversion
        result = [r async for r in self.plugin.edit_image(self.event, prompt)]
        self.assertNotIn('失败', result[0])
        self.assertEqual(self.plugin._start_edit_task.call_args.args[1], '将图一角色换成图二 保留背景')
        self.assertEqual(self.plugin._start_edit_task.call_args.args[-1], '1008x784')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.server.received = (self.headers, self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(500 if self.path == '/error' else 200)
        self.end_headers()
        if self.path == '/invalid':
            self.wfile.write(b'not json')
        elif self.path == '/url':
            self.wfile.write(json.dumps({'data': [{'url': self.server.base_url + '/image'}]}).encode())
        else:
            self.wfile.write(json.dumps({'data': [{'b64_json': base64.b64encode(b'image bytes').decode()}]}).encode())

    def do_GET(self):
        self.server.download_auth = self.headers.get('Authorization')
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'download bytes')


class RequestTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.base_url = f'http://127.0.0.1:{cls.server.server_port}'
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.plugin = Plugin(None, {'api_url': self.server.base_url,
                                    'edit_api_url': self.server.base_url, 'model': 'custom-model'})

    async def test_json_request(self):
        result = await self.plugin._generate_image('猫', 'secret', '1024x1536')
        self.assertEqual(result, b'image bytes')
        headers, body = self.server.received
        self.assertEqual(headers['Authorization'], 'Bearer secret')
        self.assertEqual(json.loads(body), {'model': 'custom-model', 'prompt': '猫', 'size': '1024x1536', 'n': 1})

    async def test_multipart_preserves_all_images(self):
        images = [(b'first', 'first.png', 'image/png'), (b'second', 'second.jpg', 'image/jpeg')]
        result = await self.plugin._edit_image('融合', 'secret', images, '1008x784')
        self.assertEqual(result, b'image bytes')
        headers, body = self.server.received
        message = BytesParser(policy=default).parsebytes(
            ('Content-Type: ' + headers['Content-Type'] + '\r\n\r\n').encode() + body)
        parts = list(message.iter_parts())
        files = [p for p in parts if p.get_param('name', header='content-disposition') == 'image[]']
        self.assertEqual([p.get_payload(decode=True) for p in files], [b'first', b'second'])
        fields = {p.get_param('name', header='content-disposition'): p.get_payload(decode=True).decode()
                  for p in parts if p.get_filename() is None}
        self.assertEqual(fields, {'model': 'custom-model', 'prompt': '融合', 'size': '1008x784', 'n': '1'})

    async def test_download_and_errors(self):
        self.plugin.config['api_url'] = self.server.base_url + '/url'
        self.assertEqual(await self.plugin._generate_image('cat', 'secret', '1024x1024'), b'download bytes')
        self.assertIsNone(self.server.download_auth)
        for path, message in [('/error', 'HTTP 500'), ('/invalid', 'JSON')]:
            self.plugin.config['api_url'] = self.server.base_url + path
            with self.assertRaisesRegex(Error, message):
                await self.plugin._generate_image('cat', 'secret', '1024x1024')

    async def test_requests_run_in_worker_thread(self):
        from threading import get_ident
        caller = get_ident()
        workers = []
        def request(*args):
            workers.append(get_ident())
            return b'result'
        self.plugin._request_image = request
        await self.plugin._generate_image('cat', 'key', '1024x1024')
        await self.plugin._edit_image('cat', 'key', [], '1024x1024')
        self.assertTrue(all(worker != caller for worker in workers))


if __name__ == '__main__':
    unittest.main()
