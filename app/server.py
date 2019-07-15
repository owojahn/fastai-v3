import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1-T7RI48z41-5hwHl0rveHBM7s2B9curM'
export_file_name = 'export.pkl'

classes = [
  'Ananas',
  'Apfel',
  'Apfelmus',
  'Apfelsaftschorle',
  'Apfelsine',
  'Aubergine',
  'Avocado',
  'Baguette',
  'Banane',
  'Bier',
  'Bier alkoholfrei',
  'Birne',
  'Blaubeeren',
  'Blumenkohl',
  'Bohnen',
  'Bonbon',
  'Braten',
  'Bratkartoffeln',
  'Bratwurst',
  'Broccoli',
  'Broetchen',
  'Butter',
  'Buttermilch',
  'Chiasamen',
  'Cola',
  'Cornflakes trocken',
  'Currywurst Pommes',
  'Diaetlimonade',
  'Eier',
  'Eiersalat',
  'Eintopf',
  'Eis',
  'Eisbergsalat',
  'Endiviensalat',
  'Erbsen gruen',
  'Erbseneintopf',
  'Erdbeeren',
  'Feldsalat',
  'Fenchel',
  'Feta Schafskaese',
  'Fisch gebraten',
  'Fisch gekocht',
  'Fisch geraeuchert',
  'Fischfrikadelle',
  'Fischkonserve',
  'Fischsalat',
  'Fischstaebchen',
  'Fleischsalat',
  'Fleischwurst',
  'Flohsamen',
  'Frikadelle',
  'Frischkaese',
  'Fruchtsaft',
  'Fruechtetee Kraeutertee',
  'Gebundene Suppe',
  'Gemischter Salat mit Dressing',
  'Gemischter Salat ohne Dressing',
  'Gemuese-Lasagne',
  'Gemueseauflauf',
  'Gemuesesaft',
  'Graubrot',
  'Gulasch Ragout',
  'Gurke',
  'Hackfleischsosse',
  'Haehnchenbrust Aufschnitt',
  'Haehnchenfleisch',
  'Haferflocken trocken',
  'Haferkleie',
  'Hamburger',
  'Himbeeren',
  'Honig',
  'Jodiertes Salz',
  'Joghurt 1,5% Fett',
  'Joghurt 3,5% Fett',
  'Kaese bis 30% Fett',
  'Kaese bis 45% Fett',
  'Kaese ueber 45% Fett',
  'Kaffee',
  'Kaffee mit Milch (Milchkaffee, Latte Macchiato)',
  'Kaffee mit Milch und Zucker (Cappucino)',
  'Kakao',
  'Kartoffelauflauf',
  'Kartoffelchips',
  'Kartoffeln',
  'Kartoffelpueree',
  'Kartoffelpuffer',
  'Kartoffelsalat',
  'Kekse',
  'Kiwi',
  'Klare Suppe',
  'Knaecke',
  'Knoedel',
  'Knuspermuesli Schokomuesli',
  'Kochschinken',
  'Kochwurst',
  'Kohlrabi',
  'Kokosmilch',
  'Kompott',
  'Kopfsalat Blattsalat',
  'Kotelett',
  'Krustentiere',
  'Kuechenkraeuter',
  'Kuerbiskerne',
  'Leberwurst',
  'Leinoel',
  'Leinsamen',
  'Likoere',
  'Limonade',
  'Linsen gekocht',
  'Linseneintopf',
  'Magerquark',
  'Mais aus Konserve',
  'Mandarine',
  'Mandelmilch',
  'Mango',
  'Mangold',
  'Margarine',
  'Margarine halbfett',
  'Marmelade Gelee',
  'Matjes',
  'Melone',
  'Mett',
  'Mineralwasser',
  'Moehren',
  'Mozzarella',
  'Muesli trocken',
  'Muesliriegel',
  'Multivitaminsaft',
  'Naturreis gekocht',
  'Nudeln gekocht',
  'Nuesse',
  'Nussnougatcreme',
  'Obstkuchen',
  'Olivenoel Rapsoel',
  'Paprika',
  'Pfannkuchen',
  'Pfirsich',
  'Pilze gegart',
  'Pizza',
  'Pommes frites',
  'Porree',
  'Pralinen',
  'Pudding',
  'Putenschnitzel',
  'Radler Alster',
  'Ratatouille',
  'Reis gekocht',
  'Rollmops',
  'Rosenkohl',
  'Rotkohl',
  'Ruehrei (2 Eier)',
  'Sahnesosse',
  'Sahnetorte Cremetorte',
  'Salami Mettwurst',
  'Salzige Knabbereien',
  'Sauerkraut',
  'Schlagsahne',
  'Schnitzel',
  'Schokolade',
  'Schokoriegel',
  'Schwarzwurzeln',
  'Sekt',
  'Sojajoghurt natur',
  'Sojamilch natur',
  'Sojaoel Erdnussoel',
  'Sojasprossen',
  'Sonnenblumenkerne',
  'Sonnenblumenoel Disteloel',
  'Sosse',
  'Spaghetti in Tomatensosse',
  'Spargel',
  'Speck Bauchfleisch',
  'Speisequark',
  'Spiegelei',
  'Spinat',
  'Spirituosen',
  'Stachelbeeren',
  'Steak',
  'Stueckchen Teilchen',
  'Suppen',
  'Tee',
  'Toast',
  'Tofu',
  'Tomate',
  'Tomatensosse',
  'Trinkmilch 1,5% Fett',
  'Trinkmilch 3,5% Fett',
  'Trockenkuchen',
  'Trockenobst',
  'Vegetarische Pizza',
  'Vollkornbroetchen',
  'Vollkornbrot',
  'Vollkornnudeln gekocht',
  'Wein',
  'Weingummi',
  'Weintrauben',
  'Weissbrot',
  'Weisskohl',
  'Wiener Wuerstchen Bockwurst',
  'Wirsing',
  'Zucchini',
  'Zucker',
  'Zwieback',
  'Zwiebeln']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    # prediction = learn.predict(img)[0]
    # return JSONResponse({'result': str(prediction)})
    pred_class,pred_idx,outputs = learn.predict(img)
    idxnew = np.argsort(outputs)
    cat = classes
    i = 1
    while outputs[idxnew[-i].item()].item() > 0.05 and i < 5:
      wkeit = round(100*outputs[idxnew[-i].item()].item(),2)
      prediction = cat[idxnew[-i].item()]
      return JSONResponse({'result': str(prediction),'wkeit': str(wkeit)})
      i +=1


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
