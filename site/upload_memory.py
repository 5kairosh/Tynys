"""Keep accepted multipart uploads in RAM and cap total request size."""
import json
from starlette.formparsers import MultiPartParser

MAX_BODY=128*1024*1024  # Ten 12 MiB files plus multipart overhead.
MultiPartParser.spool_max_size=MAX_BODY+1


class UploadMemoryLimit:
    def __init__(self,app):
        self.app=app

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('method')!='POST':
            return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]))
        try:
            length=int(headers.get(b'content-length',b'0'))
        except ValueError:
            length=0
        async def reject():
            data=json.dumps({'detail':'Сұрау тым үлкен. Әр файл 12 МБ-тан, сурет саны 10-нан аспауы керек.'},ensure_ascii=False).encode()
            await send({'type':'http.response.start','status':413,'headers':[(b'content-type',b'application/json; charset=utf-8')]})
            await send({'type':'http.response.body','body':data})
        if length>MAX_BODY:
            return await reject()
        # Buffer the bounded body before the multipart parser sees it. No tempfile rollover.
        body=bytearray()
        while True:
            event=await receive()
            if event['type']=='http.disconnect':
                return
            body.extend(event.get('body',b''))
            if len(body)>MAX_BODY:
                return await reject()
            if not event.get('more_body',False):
                break
        delivered=False
        async def buffered_receive():
            nonlocal delivered
            if not delivered:
                delivered=True
                return {'type':'http.request','body':bytes(body),'more_body':False}
            return await receive()
        await self.app(scope,buffered_receive,send)
