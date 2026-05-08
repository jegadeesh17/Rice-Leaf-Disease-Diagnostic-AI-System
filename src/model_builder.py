import keras
from keras import layers, Sequential
from keras.applications import MobileNetV2, EfficientNetB0

def build_transfer_model(base_model_class, num_classes, img_shape=(224, 224, 3), augmentation_layer=None):
    """Constructs a transfer learning model with a specified backbone."""
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=img_shape)
    base_model.trainable = False

    layers_list = [layers.Input(shape=img_shape)]
    if augmentation_layer:
        layers_list.append(augmentation_layer)
    
    layers_list.extend([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return Sequential(layers_list)
