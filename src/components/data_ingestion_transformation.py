
from src.utils import *
import shutil
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('imageclassifier','train')
    test_data_path: str=os.path.join('imageclassifier','test')
    val_data_path: str=os.path.join('imageclassifier','val')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion phase!!!")
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus: 
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info('GPU acceleration initiated...')

            data_path= os.path.join(os.getcwd(),'notebook','images')
            logging.info('Created Data Path...')

            data=tf.keras.utils.image_dataset_from_directory(data_path,image_size=(256,256),batch_size=16)
            logging.info('Read the dataset...')

            scaled_data = data.map(lambda x,y: (x/255, y))
            logging.info('Scaled the dataset...')
            train_size = int(len(scaled_data)*.7)
            val_size = int(len(scaled_data)*.2)+1
            test_size = int(len(scaled_data)*.1)+1
            train = scaled_data.take(train_size)
            val = scaled_data.skip(train_size).take(val_size)
            test = scaled_data.skip(train_size+val_size).take(test_size)
            logging.info('Splited scaled data into train, test and val...')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Save train, test, and val datasets into the specified folders
            shutil.rmtree(self.ingestion_config.train_data_path, ignore_errors=True)
            shutil.rmtree(self.ingestion_config.test_data_path, ignore_errors=True)
            shutil.rmtree(self.ingestion_config.val_data_path, ignore_errors=True)

            tf.data.Dataset.save(train,self.ingestion_config.train_data_path)
            tf.data.Dataset.save(test,self.ingestion_config.val_data_path)
            tf.data.Dataset.save(val,self.ingestion_config.test_data_path)

            logging.info('Saved train, test, and val datasets...')

            logging.info("Ingestion and transformation of the data is completed !!!")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.val_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train,test,val=obj.initiate_data_ingestion()

    modeltrainer=ModelTrainer()
    print(modeltrainer.train_model(train,val))