from bottle import Bottle, response, request
from poem_searcher import PoemsSearcher
import json
import traceback

app = Bottle()
searcher = PoemsSearcher.initialize(
    poems_pattern=[
        'data/poetry/全唐诗/唐诗三百首.json',
        'data/poetry/宋词/宋词三百首.json',
        #'data/poetry/诗经/shijing.json',
        'data/poetry/水墨唐诗/shuimotangshi.json',
        'data/poetry/曹操诗集/caocao.json',
    ],
    index_dir='data/indexes',
    force_rebuild=True
)

@app.route('/api/search', method='GET')
def search():
    query = request.query.get('q', '')
    top_k = int(request.query.get('top_k', '20'))
    threshold = float(request.query.get('threshold', '0.1'))
    
    if not query:
        response.status = 400
        return {'error': '请提供搜索词'}
    
    try:
        results = searcher.search(
            query=query,
            top_k=top_k,
            threshold=threshold
        )
        
        return {
            'code': 0,
            'data': results,
            'debug': {
                'query': query,
                'total_results': len(results),
                'top_k': top_k,
                'threshold': threshold
            }
        }
    except Exception as e:
        response.status = 500
        return {'error': str(e), 'traceback': str(traceback.format_exc())}

@app.route('/api/debug', method='GET')
def debug():
    query = request.query.get('q', '')
    if not query:
        response.status = 400
        return {'error': '请提供搜索词'}
    
    try:
        results = searcher.search(query=query)
        print(results)
        return {
            'code': 0,
            'data': {
                'type': str(type(results)),
                'sample': str(results[0]) if results else None
            }
        }
    except Exception as e:
        response.status = 500
        return {'error': str(e)}

@app.route('/', method='GET')
def home():
    return '''
    <html>
        <body>
            <h1>诗词检索服务</h1>
            <form action="/api/search" method="get">
                <textarea type="text" name="q" placeholder="输入搜索词..."></textarea>
                <button type="submit">搜索</button>
            </form>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True) 