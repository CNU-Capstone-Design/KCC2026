"""
FFHQ-style aligned CelebA 데이터셋.
preprocess_celeba.py로 전처리된 PNG 파일 로드.
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class AlignedFaceDataset(Dataset):
    def __init__(self, data_root, img_size=256, split_ratio=0.95, train=True):
        all_imgs = sorted(Path(data_root).glob('*.png'))
        n     = len(all_imgs)
        split = int(n * split_ratio)
        self.imgs = all_imgs[:split] if train else all_imgs[split:]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.imgs[idx]).convert('RGB'))


def get_dataloader(data_root, img_size, batch_size, num_workers, train=True):
    dataset = AlignedFaceDataset(data_root, img_size, train=train)
    print(f'Dataset: {len(dataset)} images ({"train" if train else "test"})')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
