import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

# ✅ Confirm eager execution
print("Eager execution:", tf.executing_eagerly())

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # ✅ Re-compile the model with correct loss function
        self.model.compile(
            loss='sparse_categorical_crossentropy',  # ✅ Fixed: use sparse version
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )

    def train_valid_generator(self):
        """
        Use image_dataset_from_directory which outputs integer labels.
        """
        self.train_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size
        )

        self.valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size
        )

        # ✅ Performance boost with prefetching
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_generator = self.train_generator.prefetch(buffer_size=AUTOTUNE)
        self.valid_generator = self.valid_generator.prefetch(buffer_size=AUTOTUNE)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        try:
            self.model.fit(
                self.train_generator,
                validation_data=self.valid_generator,
                epochs=self.config.params_epochs
            )
        except Exception as e:
            print(f"[ERROR] Training failed due to: {e}")
            raise

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
