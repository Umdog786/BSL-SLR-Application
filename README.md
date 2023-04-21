# Sign Language Recognition

This repository contains the code for a sign language recognition application. Please note that the pre-trained models used in this application are not included due to their large size. However, the training programs can be found in the `training` directory, which allows third parties to train models on any other sign language datasets.

## Getting Started

To get the application into a functioning state, you will need to train the models and obtain an I3D model. The I3D model used for this project was trained using the architecture found at [gulvarol/bsldict](https://github.com/gulvarol/bsldict). You can either source an I3D model or write a new architecture from scratch.
1. Source a sign language dataset
2. Use the `SplitDatset.py` program to split the dataset based on the identities of signers in the dataset
3. Use the `KeywordSpotting.py` program to spot keywords from the dataset and save them as video clips
4. Train a model on the dataset using any of the programs in the `training` directory
5. Load the model into `HandTracking.py` program to run the application
## Model Performance

The models produced by this project achieved the following results:

| Model Type                                             | Test Accuracy | Validation Accuracy |
|---------------------------------------------------------|---------------|---------------------|
| 3-Dimensional Convolutional Neural Network (reduced training dataset) | 83.3%        | 59.8%               |
| I3D CNN (without MediaPipe)                            | 59.8%         | 53.81%              |
| ResNet50                                                | 41.7%         | 36.7%               |
| Support Vector Machine                                  | 61.1%         | 2.6%                |

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
