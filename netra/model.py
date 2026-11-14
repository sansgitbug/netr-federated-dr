"""The one model definition shared by server, clients and dashboard."""
import numpy as np

from .config import FREEZE_BATCHNORM, IMAGE_SHAPE, LEARNING_RATE, NUM_CLASSES, REFERABLE_FROM


def build_model(pretrained: bool = False):
    """MobileNetV2 + global pooling + 5-way softmax (DR grades 0-4)."""
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE + (3,),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    if FREEZE_BATCHNORM:
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=IMAGE_SHAPE + (3,)),
            # Callers pass pixels in [0, 1]; MobileNetV2's ImageNet weights expect [-1, 1].
            tf.keras.layers.Rescaling(2.0, offset=-1.0),
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def preprocess(pil_image) -> np.ndarray:
    """PIL image -> float32 batch of one in [0, 1], shape (1, H, W, 3)."""
    img = pil_image.convert("RGB").resize(IMAGE_SHAPE)
    arr = np.asarray(img, dtype="float32") / 255.0
    return arr[np.newaxis, ...]


def decode(probs):
    """Probability vector -> (grade, confidence 0-1, referable flag)."""
    probs = np.asarray(probs).reshape(-1)
    grade = int(np.argmax(probs))
    return grade, float(probs[grade]), grade >= REFERABLE_FROM


def quadratic_weighted_kappa(cm) -> float:
    """QWK from a (K, K) confusion matrix (rows = true, cols = predicted)."""
    cm = np.asarray(cm, dtype="float64")
    k = cm.shape[0]
    n = cm.sum()
    if n == 0:
        return 0.0
    idx = np.arange(k)
    w = (idx[:, None] - idx[None, :]) ** 2 / float((k - 1) ** 2)
    expected = np.outer(cm.sum(1), cm.sum(0)) / n
    denom = (w * expected).sum()
    if denom == 0:
        return 0.0
    return float(1.0 - (w * cm).sum() / denom)


def referable_sens_spec(cm):
    """Sensitivity and specificity for referable DR (grade >= REFERABLE_FROM)."""
    cm = np.asarray(cm, dtype="float64")
    r = REFERABLE_FROM
    tp = cm[r:, r:].sum()
    fn = cm[r:, :r].sum()
    tn = cm[:r, :r].sum()
    fp = cm[:r, r:].sum()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return float(sens), float(spec)
