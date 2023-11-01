from datasets import FolderStoryDataset, FolderImageDataset, FolderStoryDatasetOri, FolderImageDatasetOri
import torch
import torchvision.transforms as transforms
import PIL
from vfid_score import fid_score as vfid_score
from fid_score_v import fid_score
#from dataset import vist as data
import functools


if __name__ == "__main__":

    with open('fid_score2.csv', 'a') as f:
        f.write('epoch,fid,vfid\n')

    image_transforms = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    gen_path = 'dataset/cmota_reproduce/test_gen'
    ori_path = 'dataset/cmota_reproduce/test_ori'

    def video_transform(video, image_transform):
        vid = []
        for im in video:
            vid.append(image_transform(im))
        vid = torch.stack(vid).permute(1, 0 ,2, 3)

        return vid

    video_transforms = functools.partial(video_transform, image_transform=image_transforms) # Only need to feed video later

    ref_dataloader = FolderStoryDatasetOri(ori_path, video_transforms)
    gen_dataloader = FolderStoryDataset(gen_path , video_transforms)

    vfid = vfid_score(ref_dataloader, gen_dataloader, cuda=True, normalize=True, r_cache=None)

    ref_dataloader = FolderImageDatasetOri(ori_path, image_transforms)
    gen_dataloader = FolderImageDataset(gen_path, image_transforms)

    fid = fid_score(ref_dataloader, gen_dataloader, cuda=True, normalize=True, r_cache=None)

    with open('fid_score2.csv', 'a') as f:
        f.write('{},{}\n'.format(fid, vfid))
