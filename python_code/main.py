try:
  import fastbook
  fastbook.setup_book()
except:
  import os
  os.sysetm("pip install -Uqq fastbook fastai")
  import fastai
  from fastai.vision.all import *
  import fastbook
  fastbook.setup_book()

print(
    "         ^           --------- \n"
    "        / \              |     \n"
    "       /   \             |     \n"
    "      /     \            |     \n"
    "     /-------\           |     \n"
    "    /         \          |     \n"
    "   /           \     --------- \n\n\n"
    "          By Quinn I            ")

try:
  path = Path('/content/gdrive/My Drive/Datasets/MurderHornetAI')
  learn_inf = load_learner(path/'export.pkl')
except:
  from fastbook import *
  from fastai.vision.widgets import *

  path = Path('/content/gdrive/My Drive/Datasets/MurderHornetAI')
  #pathOther = Path('/content/gdrive/My Drive/Datasets/Other')
  path2 = Path('/content/gdrive/My Drive/Datasets/MurderHornetAI/MurderHornet/Murder Hornet (1).jpg')

  item_tfms=Resize(128, ResizeMethod.Squish)
  item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros')
  item_tfms=RandomResizedCrop(128, min_scale=0.3)# - 30% of the image area is zoomed by specifying 0.3

  tfms = aug_transforms(do_flip = True, flip_vert = False, mult=2.0)

  data = ImageDataLoaders.from_folder(path,train = "train", valid_pct=0.2, item_tfms=Resize(128), batch_tfms=tfms, bs = 30, num_workers = 4)

  Data = DataBlock( blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, 
  splitter=RandomSplitter(valid_pct=0.2, seed=42), get_y=parent_label, item_tfms=Resize(128))
  dls = Data.dataloaders(path)

  learn = cnn_learner(data, resnet56 , metrics=error_rate)
  print("Training, this may take a minute")
  learn.fit_one_cycle(4)

  interp = ClassificationInterpretation.from_learner(learn)
  interp.plot_confusion_matrix()
  cleaner = ImageClassifierCleaner(learn)
  cleaner
  learn.export()
  
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
        import os
        os.system("pip install geocoder")
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
