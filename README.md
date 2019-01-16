# keras dataloader

DataLoader for keras

## Usage example

```python

from keras_dataloader.dataloader import DataGenerator
from keras_dataloader.dataset import Dataset


class TensorDataset(Dataset):

    def __getitem__(self, index):
        # time.sleep(np.random.randint(1, 3))
        return np.random.rand(3), np.array([index])

    def __len__(self):
        return 100
        
model = Sequential()
model.add(Dense(units=4, input_dim=3))
model.add(Dense(units=1))
model.compile('adam', loss='mse')

data_loader = DataGenerator(TensorDataset(), batch_size=20, num_workers=0)

model.fit_generator(generator=data_loader, epochs=1, verbose=1)

```