import torch
import os
import pickle

def collate_tensors(batch):
    out = None
    elem = batch[0]
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    return torch.stack(batch, 0, out=out)

def collate_fn(batch):
    tensors = [t for item in batch for t in item['tensor']]
    labels = [t for item in batch for t in item['label']]

    return collate_tensors(tensors), torch.FloatTensor(labels).unsqueeze(1)

def create_ckpt_folder(checkpoints_folder):
    # Saving the current split for testing later on.
    count = 0
    for f in os.listdir(checkpoints_folder):
        if 'checkpoint' in f:
            count += 1
    ckpt_folder = f"checkpoint{count}"
    os.mkdir(checkpoints_folder + ckpt_folder)
    print(f"Current folder is: {count}")
    return ckpt_folder

# returns folder where split is saved.
def pickle_datasets(root_dir, trainset_augment, trainset, valset, testset, source_sets=None):
    # Saving the current split for testing later on.
    checkpoints_folder = root_dir + 'checkpoints/'
    ckpt_folder = create_ckpt_folder(checkpoints_folder)
    with open(checkpoints_folder + ckpt_folder + '/train_augment.pkl', 'wb') as f:
        pickle.dump(trainset_augment, f)
    with open(checkpoints_folder + ckpt_folder + '/train.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open(checkpoints_folder + ckpt_folder + '/val.pkl', 'wb') as f:
        pickle.dump(valset, f)
    with open(checkpoints_folder + ckpt_folder + '/test.pkl', 'wb') as f:
        pickle.dump(testset, f)
    if source_sets:
        with open(checkpoints_folder + ckpt_folder + '/source_sets.pkl', 'wb') as f:
            pickle.dump(source_sets, f)
    return ckpt_folder

def pickle_dataset(root_dir, dataset, ckpt_num):
    checkpoint_folder = f"{root_dir}checkpoints/checkpoint{str(ckpt_num)}"
    with open(f"{checkpoint_folder}/separated_datasets.pkl", 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    root_dir = '/scratch/groups/kpohl/transformers_mri/'
    count = 0 
    checkpoint_folder = root_dir + 'checkpoints/'
    for f in os.listdir(checkpoint_folder):
        if 'checkpoint' in f:
            count += 1
    print('Current folder is: ', count)