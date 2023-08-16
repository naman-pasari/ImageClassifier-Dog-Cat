from src.utils import *

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("imageclassifier","imageclassifier.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def train_model(self,train_data_path, val_data_path):
        try:
            logging.info("Intilizing Model training process...")
            train = tf.data.experimental.load(train_data_path)
            val = tf.data.experimental.load(val_data_path)
            logging.info("Data loaded successfully...")
            base_model= tf.keras.applications.MobileNetV2(input_shape=(256,256,3),
                                              include_top=False,
                                              weights='imagenet',
                                              classes=2
            )
            base_model.trainable = False
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1)
            ])
            logging.info("Model created successfully...")

            model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
            model.fit(train,epochs=10,validation_data=val)
            logging.info("Model compiled successfully...")

            # Save the trained model as "imageclassifier.h5"
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info("Model saved as imageclassifier.h5")

            return self.model_trainer_config.trained_model_file_path
       
        except Exception as e:
            raise CustomException(e,sys)
