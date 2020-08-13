#API_SERVICE

import os
### Telegram Dependence ###
from telegram.ext import CommandHandler, Filters, MessageHandler, Updater
from argparse import ArgumentParser

### Machine Learning Dependence ###
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions 
###############################
import numpy as np

TELEGRAM_TOKEN = "1341613930:AAGnBo-sPAd4tQQOSzm2AhaumHoyQeHYfpM"

def start(bot, update):
    response_message = "Bem-vindo ao Bot de Reconhecimento de Imagens - Disciplina Redes de Computadores \n Digite a opção desejada:\n /info - Informações dos Desenvolvedores " + "\n /reconhecedor - Modulo para reconhecer objetos"
    bot.send_message(chat_id=update.message.chat_id, text=response_message)

def info(bot, update):
    response_message = "Este APP foi desenvolvido pela equipe: Caio Falcão, Ricardo Marques e Weslley Kelson"
    bot.send_message(chat_id=update.message.chat_id, text=response_message)

    
def get_photo(update):  
    downloads_dir = 'downloaded_images'
    os.makedirs(downloads_dir, exist_ok=True)
    temp_path = os.path.join(downloads_dir,'file_%s_id_%d_temp.png' % (update.message.photo[-1].file_id, update.message.message_id))
    update.message.photo[-1].get_file().download(temp_path)
    return temp_path

def reconhecedor(bot, update):
    response_message = "\n Tire a foto de um objeto (Ex: Xícara, Bola, Óculos, Isqueiro...) "
    bot.send_message(chat_id=update.message.chat_id, text=response_message)

def imageRecognizer(bot, update):
    
    temp_path = get_photo(update)
    resultPredict = classifierImage(temp_path)
   
    response_message = resultPredict
    bot.send_message(chat_id=update.message.chat_id, text=response_message)
    reconhecedor(bot,update)
    
def classifierImage(pathImage):
    # load the model
    model = ResNet50(weights='imagenet')
    img = image.load_img(pathImage, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    label = decode_predictions(preds)
    label = label[0][0]
  
    # print the classification
    result = '%s (%.2f%%)' % (label[1], label[2]*100) 
    return result

def main():
    

    # Create the Updater and pass it your bot's token.
    updater = Updater(token=TELEGRAM_TOKEN)
    
    
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("info", info))
    dispatcher.add_handler(CommandHandler("reconhecedor", reconhecedor))
    dispatcher.add_handler(MessageHandler(Filters.photo, imageRecognizer))
    
    
    # Start the Bot    
    
    updater.start_polling()
    updater.idle()
    


if __name__ == "__main__":
    
    print("press CTRL + C to cancel.")
    main()
