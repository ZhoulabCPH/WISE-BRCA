import feather
import torch

from torch.utils.data import Dataset


def collate(batch):
    patches_features_224 = [b['patches_features_224'] for b in batch]
    patches_names_224 = [b['patches_names_224'] for b in batch]
    patches_features_512 = [b['patches_features_512'] for b in batch]
    patches_names_512 = [b['patches_names_512'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch])

    id = [b['id'] for b in batch]
    patches_features_224 = torch.tensor(patches_features_224, dtype=torch.float)
    patches_features_512 = torch.tensor(patches_features_512, dtype=torch.float)

    return {'patches_features_512': patches_features_512, 'patches_names_512': patches_names_512,
            'patches_features_224': patches_features_224, 'patches_names_224': patches_names_224,'labels': labels, 'id': id}


class BC(Dataset):
    def __init__(self, workspace, slides_path_224, slides_path_512):
        super(BC, self).__init__()
        self.workspace = workspace
        self.slides_path_224 = slides_path_224
        self.slides_path_512 = slides_path_512

        self.slides = list(self.workspace.loc[:, 'slides'])

    def __len__(self):
        return len(self.slides)

    def load_patches_features(self, slide_name):
        slide_name = slide_name.replace('csv', 'feather')
        slide_path_224 = f'{self.slides_path_224}/{slide_name}'
        slide_224 = feather.read_dataframe(slide_path_224)
        slide_path_512 = f'{self.slides_path_512}/{slide_name}'
        slide_512 = feather.read_dataframe(slide_path_512)
        patches_names_224 = slide_224.index
        patches_names_512 = slide_512.index
        patches_features_224 = slide_224.iloc[:, 0:].values
        patches_features_512 = slide_512.iloc[:, 0:].values
        return patches_names_224, patches_features_224, patches_names_512, patches_features_512

    def __getitem__(self, item):
        label = self.workspace.iloc[item]['BRCA_mut']
        slide_name = self.workspace.iloc[item]['slides']
        id = slide_name
        patches_names_224, patches_features_224, patches_names_512, patches_features_512 = self.load_patches_features(slide_name)
        return {'patches_features_224': patches_features_224, 'patches_names_224': patches_names_224,
                'patches_features_512': patches_features_512, 'patches_names_512': patches_names_512,'label': label, 'id': id}



