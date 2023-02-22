import numpy as np
from matplotlib import pyplot as plt

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv
import torch
from tools.maskclip_utils.prompt_engineering import zeroshot_classifier, prompt_templates
import random
from clip_text_decoder.model import ImageCaptionInferenceModel
from sentence_transformers.util import community_detection
from skimage.segmentation import slic
from PIL import Image
from prefix_cap2 import caption_from_CLIP, ClipCaptionModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from matplotlib import cm
import clip
import matplotlib.patches as mpatches

from bokeh.plotting import figure, show, gridplot
from bokeh.models import ColumnDataSource, HoverTool, CustomJS
def merge_clusters(slic_clusters, threshold=0.995):
    cluster_sizes = []
    cluster_embs = []

    for i in range(np.max(slic_clusters)):
        mask = np.broadcast_to(slic_clusters == i, clip_map.shape)
        size = np.count_nonzero(mask)
        avg_embedding = np.mean(np.ma.masked_array(clip_map, mask), axis=(1, 2))
        cluster_embs.append(avg_embedding)
        cluster_sizes.append(size)

        avg_embedding = torch.tensor(avg_embedding).float().to(device)


    similarities = cosine_similarity(cluster_embs)
    for i in range(len(similarities)):
        similarities[i, i] = 0
    most_similar = np.unravel_index(similarities.argmax(), similarities.shape)

    while similarities[most_similar] > threshold:
        print("Similarity: ", similarities[most_similar])
        print("Merging clusters")
        slic_clusters[slic_clusters == most_similar[1]] = most_similar[0]
        for i in range(most_similar[1], np.max(slic_clusters)):
            slic_clusters[slic_clusters == i + 1] = i

        cluster_embs[most_similar[0]] = cluster_embs[most_similar[0]] * cluster_sizes[most_similar[0]] \
                                        + cluster_embs[most_similar[1]] * cluster_sizes[most_similar[1]] \
                                        / (cluster_sizes[most_similar[0]] + cluster_sizes[
            most_similar[1]])  # weighted average
        cluster_sizes[most_similar[0]] += cluster_sizes[most_similar[1]]
        cluster_embs.pop(most_similar[1])
        cluster_sizes.pop(most_similar[1])

        similarities = cosine_similarity(cluster_embs)
        for i in range(len(similarities)):
            similarities[i, i] = 0
        most_similar = np.unravel_index(similarities.argmax(), similarities.shape)

    return slic_clusters

def get_masked_clip_embeddings(model, image, clusters):
    clusters = Image.fromarray(clusters.astype(np.uint8))
    clusters = clusters.resize((7, 7), resample=Image.NEAREST)
    clusters = np.array(clusters)
    embeddings = []
    for i in range(np.max(clusters)+1):
        model.visual.mask = np.where(clusters.flatten()==i)[0]
        mask_embedding = model.encode_image(image)
        embeddings.append(mask_embedding)
    return embeddings
def cluster_captions(clip_map, clusters, model):
    captions = []
    for i in range(np.max(clusters)+1):
        mask = np.broadcast_to(clusters == i, clip_map.shape)
        avg_embedding = np.mean(np.ma.masked_array(clip_map, mask), axis=(1,2))
        avg_embedding = torch.tensor(avg_embedding).float().to(device)
        caption = caption_from_CLIP(avg_embedding, model)
        captions.append(caption)
    return captions

config_file = '../configs/maskclip/maskclip_vit16_512x512_demo.py'
config = mmcv.Config.fromfile(config_file)
checkpoint_file = '../pretrain/ViT16_clip_backbone.pth'

img_path = 'cat.jpg'
fg_classes = ['person', 'window', 'sofa', 'table']
bg_classes = ['wall', 'floor']

text_embeddings = zeroshot_classifier('ViT-B/16', fg_classes+bg_classes, prompt_templates)
text_embeddings = text_embeddings.permute(1, 0).float()
torch.save(text_embeddings, '../pretrain/demo_ViT16_clip_text.pth')

num_classes = len(fg_classes + bg_classes)
config.model.decode_head.num_classes = num_classes
config.model.decode_head.text_categories = num_classes

config.data.test.fg_classes = fg_classes
config.data.test.bg_classes = bg_classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# build the model from a config file and a checkpoint file
model = init_segmentor(config, checkpoint_file, device='cuda:0')

backbone = model.backbone
head = model.decode_head

#get embedding of the image with CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
image_clip = clip_model.encode_image(image).detach().cpu().numpy()

# test a single image
clip_map = inference_segmentor(model, img_path)[0]
del model, backbone, head

img = np.array(Image.open(img_path))

slic_clusters = slic(clip_map, n_segments=8, compactness=0.001, sigma=1, channel_axis=0, enforce_connectivity=False)

print(np.unique(slic_clusters))
slic_clusters = merge_clusters(slic_clusters, threshold=0.997)
print(np.unique(slic_clusters))


mask_embeddings = get_masked_clip_embeddings(clip_model, image, slic_clusters)
mask_embeddings_array = np.vstack([emb.detach().cpu().numpy() for emb in mask_embeddings])

del clip_model

prefix_length = 10
caption_model = ClipCaptionModel(prefix_length)
caption_model_path = '../pretrain/coco_weights_cap.pt'
caption_model.load_state_dict(torch.load(caption_model_path, map_location=device))
caption_model = caption_model.eval()
caption_model = caption_model.to(device)
full_caption = caption_from_CLIP(torch.tensor(image_clip), caption_model)
print("Full image caption: ", full_caption)

mask_captions = []
print("Masked captions: ")
for mask_emb in mask_embeddings:
    mask_emb = torch.tensor(mask_emb).float().to(device)
    caption = caption_from_CLIP(mask_emb, caption_model)
    mask_captions.append(caption)
    print(caption)

#t-sne visualization
flat_embeddings = clip_map.reshape((*clip_map.shape[:-2], -1))
flat_embeddings = np.hstack((flat_embeddings, image_clip.T, mask_embeddings_array.T))
tsne = TSNE(n_components=2).fit_transform(flat_embeddings.T)

palette = cm.get_cmap('viridis', np.max(slic_clusters)+1)
colors = [palette(i) for i in list(slic_clusters.flatten())] #+ [(1,0,0)] + [palette(i) for i in range(np.max(slic_clusters)+1)]
markers = ['.'  for i in list(slic_clusters.flatten())] + ['o'] + ['x' for i in range(np.max(slic_clusters)+1)]
plt.scatter(tsne[:-len(mask_embeddings)-1, 0], tsne[:-len(mask_embeddings)-1, 1], c=colors, marker='.')
plt.scatter(tsne[-len(mask_embeddings), 0], tsne[-len(mask_embeddings), 1], c='r', marker='o')
colors = [palette(i) for i in range(np.max(slic_clusters)+1)]
plt.scatter(tsne[-len(mask_embeddings):, 0], tsne[-len(mask_embeddings):, 1], c=colors, marker='x')
plt.show()

embedding_captions = cluster_captions(clip_map, slic_clusters, caption_model)
print("Average cluster captions:")
print(embedding_captions)

fig, axs = plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [1, 2]})
im = plt.imread(img_path)
implot = axs[0].imshow(im)
axs[0].set_title(full_caption)
cluster_plot = axs[1].imshow(slic_clusters)
values = np.unique(slic_clusters.ravel())

colors = [ cluster_plot.cmap(cluster_plot.norm(value)) for value in values]
# create a patch (proxy artist) for every color
patches = [ mpatches.Patch(color=colors[i], label=mask_captions[i] ) for i in range(len(values)) ]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )


#plt.imshow(clustered_img)
plt.tight_layout()
plt.show()
# show the results
#show_result_pyplot(model, img, result, None)

