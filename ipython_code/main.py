!wget 'https://www.dropbox.com/s/ivuiy5pekyndk81/export.pkl?dl=0' && mv 'export.pkl?dl=0' 'export.pkl' && rm 'export.pkl?dl=0.1'
try:
  import fastbook
  #fastbook.setup_book()
  import fastai
except:
  !pip install -Uqq fastbook fastai
  import fastai
  from fastai.vision.all import *
  import fastbook
  #fastbook.setup_book()
print(
    "         ^           --------- \n"
    "        / \              |     \n"
    "       /   \             |     \n"
    "      /     \            |     \n"
    "     /-------\           |     \n"
    "    /         \          |     \n"
    "   /           \     --------- \n\n\n"
    "          By Quinn I            ")
from fastbook import *
from fastai.vision.all import *
fastbook.setup_book()
path = Path('./')
from fastai.vision.widgets import *
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()
btn_run = widgets.Button(description='Classify')
def on_click_classify(change):
  img = PILImage.create(btn_upload.data[-1])
  out_pl.clear_output()
  with out_pl: display(img.to_thumb(128,128))
  pred,pred_idx,probs = learn_inf.predict(img)
  #lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
  if f'Prediction: {pred}' == 'Prediction: MurderHornet':
    prob = f'{probs[pred_idx]:.04f}'
    prob = float(prob)
    if prob >= 0.85:
      print('IS HORNET SENDING EMAIL')
      try:
        import geocoder
      except:
        !pip install geocoder
        import geocoder
      g = geocoder.ip('me')
      print(g.latlng)
  else:
    print(f"Classified as {pred} with {probs[pred_idx]:.04f} confidence (out of one)")
learn_inf = load_learner(path/'export.pkl')
btn_run.on_click(on_click_classify)
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()
print('\n\n\nSelect an image')
VBox([widgets.Label(''),btn_upload, btn_run, out_pl, lbl_pred])
