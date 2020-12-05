import input_pipeline
import tensorflow as tf
import pathlib
import tensorflow_datasets as tfds


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image

def crate_dataset(path):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data_root = pathlib.Path('./data')
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    label_names = sorted(
        item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index)
                          for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image,
                           num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(all_image_labels, tf.int64))
    #image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    image_label_ds = tf.data.Dataset.zip({'image': image_ds, 'label': label_ds})
    return image_label_ds

