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
from prefix_cap import init_clip_cap, caption_from_CLIP
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from matplotlib import cm
import clip
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
def cluster_captions(clip_map, clusters):
    captions = []
    model_path = '../pretrain/prefix_cap_model.pt'
    model, tokenizer = init_clip_cap(model_path, device)
    for i in range(np.max(clusters)+1):
        mask = np.broadcast_to(clusters == i, clip_map.shape)
        avg_embedding = np.mean(np.ma.masked_array(clip_map, mask), axis=(1,2))
        avg_embedding = torch.tensor(avg_embedding).float().to(device)
        caption = caption_from_CLIP(avg_embedding, model, tokenizer)
        captions.append(caption)
    return captions

config_file = '../configs/maskclip/maskclip_vit16_512x512_demo.py'
config = mmcv.Config.fromfile(config_file)
checkpoint_file = '../pretrain/ViT16_clip_backbone.pth'

img_path = 'batman.jpg'
fg_classes = ['person', 'window', 'sofa', 'table']
bg_classes = ['wall', 'floor']

text_embeddings = zeroshot_classifier('ViT-B/16', fg_classes+bg_classes, prompt_templates)
text_embeddings = text_embeddings.permute(1, 0).float()
print(text_embeddings.shape)
torch.save(text_embeddings, '../pretrain/demo_ViT16_clip_text.pth')

num_classes = len(fg_classes + bg_classes)
config.model.decode_head.num_classes = num_classes
config.model.decode_head.text_categories = num_classes

config.data.test.fg_classes = fg_classes
config.data.test.bg_classes = bg_classes

# config.model.decode_head.num_vote = 1
# config.model.decode_head.vote_thresh = 1.
# config.model.decode_head.cls_thresh = 0.5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# build the model from a config file and a checkpoint file
model = init_segmentor(config, checkpoint_file, device='cuda:0')

backbone = model.backbone
head = model.decode_head

# test a single image
clip_map = inference_segmentor(model, img_path)[0]
del model, backbone, head
img = np.array(Image.open(img_path))

#get embedding of the image with CLIP
# clip_model, preprocess = clip.load("ViT-B/16", device=device)
# image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
# image_clip = clip_model.encode_image(image).detach().cpu().numpy()

image_slic = slic(img, n_segments=5, compactness=20, sigma=1, channel_axis=2, enforce_connectivity=True)
slic_clusters = slic(clip_map, n_segments=8, compactness=0.01, sigma=1, channel_axis=0, enforce_connectivity=False)
#create a list of image patches corresponding to each token from image_slic
patch_size = (img.shape[0]//slic_clusters.shape[0], img.shape[1]//slic_clusters.shape[1])
patches = []
for i in range(len(slic_clusters.flatten())):
    x = i // slic_clusters.shape[1]
    y = i % slic_clusters.shape[1]
    patches.append(img[x*patch_size[0]:(x+1)*patch_size[0], y*patch_size[1]:(y+1)*patch_size[1]])

print(np.unique(slic_clusters))
slic_clusters = merge_clusters(slic_clusters, threshold=0.997)
print(np.unique(slic_clusters))

#t-sne visualization
flat_embeddings = clip_map.reshape((*clip_map.shape[:-2], -1))
# flat_embeddings = np.hstack((flat_embeddings, image_clip.T))
tsne = TSNE(n_components=2).fit_transform(flat_embeddings.T)
# colors = cm.get_cmap('viridis', np.max(slic_clusters)+1)
# colors = [colors(i) for i in list(slic_clusters.flatten())] + [(1,0,0)]
# plt.scatter(tsne[:, 0], tsne[:, 1], c=colors, marker='o')
# plt.show()


# Create a Bokeh plot of the t-SNE embedding
source = ColumnDataSource(data=dict(
    x=tsne[:, 0],
    y=tsne[:, 1],
))

plot = figure(tools=[HoverTool()])
plot.scatter('x', 'y', source=source)
hover = plot.select_one(HoverTool)

# Define the JavaScript code to highlight the corresponding part of the image
code = """
    var index = cb_data.index['1d'].indices[0];
    var x = source.data.x[index];
    var y = source.data.y[index];
    var img_data = img_data_source.data['image'][0];
    var img_width = img_data_source.data['image_width'][0];
    var img_height = img_data_source.data['image_height'][0];
    var img_x = Math.floor(x * img_width);
    var img_y = Math.floor(y * img_height);
    var pixel_index = (img_y * img_width + img_x) * 4;
    for (var i = 0; i < 4; i++) {
        img_data[pixel_index + i] = 255 - img_data[pixel_index + i];
    }
    img_data_source.change.emit();
"""



# Define the Bokeh data source for the image
image_data = np.flipud(np.rot90(img))
#image_data = np.dstack((image_data[:,:,0], image_data[:,:,1], image_data[:,:,2], 255*np.ones((256, 256), dtype=np.uint8)))
image_source = ColumnDataSource(data=dict(image=[image_data], image_width=[image_data.shape[1]], image_height=[image_data.shape[0]]))

# Create a JavaScript callback to handle the interactions between the t-SNE plot and the image
#callback = CustomJS(args=dict(source=source, img_data_source=img_path), code=code)
#hover.callback = callback

# Add the image to the Bokeh plot
plot_img = figure()
plot_img.image_rgba(image='image', x=0, y=0, dw='image_width', dh='image_height', source=image_source)


plot_final = gridplot([[plot_img]])
# Show the Bokeh plot
show(plot_final)

# captions = cluster_captions(clip_map, slic_clusters)
# print(captions)


#clustered_img = np.zeros(np.shape(result)[1:3])
# result = torch.from_numpy(result).to(device)
# result = torch.flatten(result, start_dim=1).permute(1, 0)
#
# result =  result / result.norm(dim=1).unsqueeze(dim=1)
# clusters = community_detection(result,
#                                min_community_size=25,
#                                threshold=0.85)
#
# w, h = clustered_img.shape
# for i, c in enumerate(clusters):
#     for patch in c:
#         clustered_img[patch%w, patch//w] = i

# caption_model = ImageCaptionInferenceModel.load("../pretrain/clip_caption_model.pt").to(device)
# points=[]
# for i in range(5):
#     w, h = result.shape[1], result.shape[2]
#     x = random.randint(0, w)
#     y = random.randint(0, h)
#     random_embedding = torch.from_numpy(result[:,x,y]).to(device)
#     caption = caption_model(random_embedding, beam_size=1)
#
#     points.append((x,y, caption))
#
#     print((x,y))
#     print(caption)


im = plt.imread(img_path)
plt.subplot(1, 3, 1)
implot = plt.imshow(im)
# for point in points:
#     plt.scatter(point[1], point[0], label = point[2])
# plt.legend(loc=(0, 1))
plt.subplot(1, 3, 2)
plt.imshow(slic_clusters)
plt.gca().set_title('CLIP clusters')

plt.subplot(1, 3, 3)
plt.imshow(image_slic)
plt.gca().set_title('Pixel clusters')

#plt.imshow(clustered_img)
plt.tight_layout()
plt.show()
# show the results
#show_result_pyplot(model, img, result, None)

