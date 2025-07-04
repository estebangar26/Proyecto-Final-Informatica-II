# Proyecto-Final-Informatica-II
Scripts de Programación Orientada a Objetos
import pydicom
import cv2
import os 
import numpy as np
from pymongo import MongoClient
from scipy import stats
import json

# Todas las funciones necesarias para contener y manipular la información necesaria para el programa. 

class Image_processing:
    def __init__(self):
        pass

    def crear_usuario(self, usuario, contraseña):
        try:
            with open("usuarios.json", "r") as archivo:
                usuarios = json.load(archivo)
        except FileNotFoundError:
            usuarios = {}

        usuarios[usuario] = contraseña

        with open("usuarios.json", "w") as archivo:
            json.dump(usuarios, archivo)

        return "Usuario {} creado exitosamente.".format(usuario)

    def validar_credenciales(self, usuario, contraseña):
        with open("usuarios.json", "r") as archivo:
            usuarios = json.load(archivo)
            if usuario in usuarios and usuarios[usuario] == contraseña:
                return True
        return False
        
    def load_folder(self, url:str, anonymize=False):
        """ 
        Función que permite cargar todos los archivos de image en una carpeta, y retornarlos 
        en una lista, además de, en el caso de los archivos dicom y si el usuario desea, crear
        una nueva carpeta con toda la información ya anonimizada.

        Args:
        url (str): URL donde se encuentra la carpeta.
        anonymize (bool): Decide si anonimizar o no los archivos DICOM.
        """
        
        images = []
        #i = 1
        for path in os.listdir(url):
            image, file = self.load_file(url + "/" + path, anonymize)
            # if file != None and anonymize:
            #     # if not os.path.exists("./Anonimizadas"):
            #     #     os.makedirs("./Anonimizadas")
            #     # file.save_as("./Anonimizadas/" + f"anon{str(i).zfill(2)}.dcm")
            #     # i += 1
            #     images.append({"name": path, "file": file, "image": image})

            if type(image) != None:    
                images.append({"name": path, "file": file, "image": image})
            
            else:
                print(f"Error: {path} no es un archivo de imagen válido.")

        return images

    def load_file(self, url:str, anonymize=False):
        """ 
        Función que carga retorna la image en una dirección particular, ya sea jpg, png o dicom.
        En este último caso, además, anonimiza, si el usuario así desesa, la información del 
        paciente y la retorna junto con la image.

        Args:
        url: url donde se encuentra el archivo. 
        """
        if url.endswith(".dcm"):
            file = pydicom.dcmread(url)
            if anonymize:
                self.anonymize(file)
            image = file.pixel_array

            return image, file
        
        elif url.endswith(".jpg") or url.endswith(".png"):
            # cv2 no lee caracteres especiales como tildes o ñ, toca hacer esto:
            with open(u""+url, 'rb') as f:
                bytes = bytearray(f.read())
            
            array = np.asarray(bytes, dtype=np.uint8)
            bgr_image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            return image, None
        
        else:
            return None, None
        
    def anonymize(self, dicom_file):
        for data_element in dicom_file.iterall():
            if data_element.tag.group == 0x0010:
                if type(data_element.value) == str:
                    data_element.value = "N/A"

                elif type(data_element.value) == int:
                    data_element.value = 0
                
    def close_open(self,image,kernel):
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return image

    def open_close(self,image, kernel):
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image

    def dicom_extract_image(self, dicom_file):
        image = dicom_file.pixel_array
        return image

    def dicom_save_image(self, dicom_file, image):
        dicom_file.PixelData = image.tobytes()
        dicom_file.Rows, dicom_file.Columns = image.shape

    def close_open_vs_open_close(self, kernel:np.array, images_url:str, anonymize=False):
        """ 
        Función que retorna una lista de imagenes, que son el resultado después de un proceso
        de apertura-cierre y otra de cierre-apertura.

        Args:
        kernel (array): Kernel con el que se hará el procesamiento.
        images_url (str): Dirección de las imágenes a procesar.
        anonymize (bool): Decide si anonimizar las imágenes o no. Por defecto False.
        """
        images = self.load_folder(images_url, anonymize)
        processed_images = []
        # i = 1
        # j = 1
        # plt.figure(figsize=(10,200))

        for image in images:
            if type(image["image"]) == pydicom.dataset.FileDataset:
                image["image"] = self.dicom_extract_image(image["image"])
            
            # plt.subplot(self.__number_of_images, 3, i)
            # plt.axis("off")
            # plt.title(f"Imagen {str(j).zfill(2)}")
            # plt.imshow(image)
            # j+=1
            # i+=1 
            processed_images.append({
                "name": image["name"],
                "file": image["file"],
                "image": image["image"]
            })

            # plt.subplot(self.__number_of_images, 3, i)
            # plt.axis("off")
            # plt.title(f"Cierre - Apertura")
            # plt.imshow(self.close_open(image, kernel))
            # i+=1 

            if image["file"] != None:
                self.dicom_save_image(image["file"], self.close_open(image["image"], kernel))

            processed_images.append({
                "name": "close_open_" + image["name"],
                "file": image["file"],
                "image": self.close_open(image["image"], kernel)
            })

            # plt.subplot(self.__number_of_images, 3, i)
            # plt.axis("off")
            # plt.title(f"Apertura - Cierre")
            # plt.imshow(self.open_close(image, kernel))
            # i+=1
            
            if image["file"] != None:
                self.dicom_save_image(image["file"], self.open_close(image["image"], kernel))

            processed_images.append({
                "name": "open_close_" + image["name"],
                "file": image["file"],
                "image": self.open_close(image["image"], kernel)
            })

        # plt.tight_layout()
        # plt.show()

        return processed_images

    def cut(self, image, x, y):
        return image[-x:, -y:]

    def cut_and_resized(self, images_url:str, x, y, anonymize=None):
        images = self.load_folder(images_url, anonymize)

        processed_images = []
        
        # j=1
        # i=1
        # plt.imshow(images[0])
        # plt.show()
        # plt.figure(figsize=(10,200))
        for image in images:
            if type(image["image"]) == pydicom.dataset.FileDataset:
                image["image"] = self.dicom_extract_image(image["image"])
            
            processed_images.append({
                "name": image["name"],
                "file": image["file"],
                "image": image["image"]
            })

            cut_image = image["image"][-x:,-y:]
            resized_image = cv2.resize(cut_image, (cut_image.shape[0]*2, cut_image.shape[1]*2))
            # cv2.imwrite(f"{resized}resized{str(j).zfill(2)}.jpg", resized_image)
            # j += 1

            # ax1 = plt.subplot(self.__number_of_images, 3, i)
            # plt.title(f"Imagen {str(j).zfill(2)}")
            # plt.imshow(image, cmap="inferno", aspect="equal")
            # i += 1

            # plt.subplot(self.__number_of_images,3,i, sharex=ax1, sharey=ax1)
            # plt.title("Imagen recortada")
            # plt.imshow(cut_image, cmap="inferno", aspect="equal")
            # i += 1
            if image["file"] != None:
                self.dicom_save_image(image["file"], cut_image)

            processed_images.append({
                "name": "cut_" + image["name"],
                "file": image["file"],
                "image": cut_image
            })

            # plt.subplot(self.__number_of_images,3,i, sharex=ax1, sharey=ax1)
            # plt.title("Imagen reescalada")
            # plt.imshow(resized_image, cmap="inferno", aspect="equal")
            # i += 1
            if image["file"] != None:
                self.dicom_save_image(image["file"], resized_image)

            processed_images.append({
                "name": "resized_" + image["name"],
                "file": image["file"],
                "image": resized_image
            })


        # plt.tight_layout()
        # plt.show()
        return processed_images

    def media(self, matriz, kernel):
        return cv2.filter2D(matriz, -1, kernel)

    def kernel(self, x,y):
        kernel= np.ones((x,y), np.uint8) / (x*y)
        return kernel

    def suavizado(self, images_url:str, anonymize=False):
        images = self.load_folder(images_url, anonymize)
        processed_images = []
               
        # plt.figure(figsize=(10,200))
        # i = 1
        # j = 1
        
        for image in images:
            if type(image["image"]) == pydicom.dataset.FileDataset:
                image["image"] = self.dicom_extract_image(image["image"])    
            
            # plt.subplot(self.__number_of, 4, i)
            # plt.axis("off")
            # plt.title(f"Imagen {str(j).zfill(2)}")
            # plt.imshow(image, cmap="inferno")
            # i += 1
            processed_images.append({
                "name": image["name"],
                "file": image["file"],
                "image": image["image"]
            })

            
            img_3X3 = self.media(image["image"], self.kernel(3,3))
            # plt.subplot(self.__number_of_images, 4, i)
            # plt.axis("off")
            # plt.title(f"Suavizado con kernel 3x3")
            # plt.imshow(img_3X3, cmap="inferno")
            # cv2.imwrite(f"{softened}suavizado{str(j).zfill(2)}.jpg", img_3X3.shape)
            # i += 1
            if image["file"] != None:
                self.dicom_save_image(image["file"], img_3X3)

            processed_images.append({
                "name": "3x3_" + image["name"],
                "file": image["file"],
                "image": img_3X3
            })

            img_5X5 = self.media(image["image"], self.kernel(5,5))
            # plt.subplot(self.__number_of_images, 4, i)
            # plt.axis("off")
            # plt.title("Suavizado con kernel 5x5")
            # plt.imshow(img_5X5, cmap="inferno")
            # cv2.imwrite(f"{softened}suavizado{str(j).zfill(2)}.jpg", img_5X5.shape)
            # i += 1
            if image["file"] != None:
                self.dicom_save_image(image["file"], img_5X5)

            processed_images.append({
                "name": "5x5_" + image["name"],
                "file": image["file"],
                "image": img_5X5
            })
            
            img_7X7= self.media(image["image"], self.kernel(7,7))
            # plt.subplot(self.__number_of_images, 4, i)
            # plt.axis("off")
            # plt.title("Suavizado con kernel 7x7")
            # plt.imshow(img_7X7, cmap="inferno")
            # cv2.imwrite(f"{softened}suavizado{str(j).zfill(2)}.jpg", img_3X3.shape)
            # i += 1
            # j +=1
            if image["file"] != None:
                self.dicom_save_image(image["file"], img_7X7)
            
            processed_images.append({
                "name": "7x7_" + image["name"],
                "file": image["file"],
                "image": img_7X7
            })

        # plt.show()
        return processed_images
    
    def pymongo_save(self, images_url, anonymize=False):
        images = self.load_folder(images_url, anonymize)

        username = "seguimiento3"
        password = "mOtHxcAnhBMSNoLN"
        cluster_uri = f"mongodb+srv://{username}:{password}@info2.hfxzxrr.mongodb.net/"
        client = MongoClient(cluster_uri)
        data_base = client["DICOM_data"]
        collection_1 = data_base["All_Data"]
        collection_2 = data_base["Experiments_Data"]

        to_save_data_1 = []
        to_save_data_2 = []

        for image in images:

            if image["name"].startswith("3X3") or image["name"].startswith("5X5") or image["name"].startswith("7X7"):
                experiment = "Suavizado"
            
            elif image["name"].startswith("close_open"):
                experiment = "Apertura y cierre"
            
            elif image["name"].startswith("open_close"):
                experiment = "Cierre y apertura"

            elif image["name"].startswith("cut"):
                experiment = "Recorte"
            
            elif image["name"].startswith("resized"):
                experiment = "Recorte y Ampliado"
            
            else:
                experiment = None
            
            metadata = {}
            if image["file"] != None:

                for attribute in dir(image["file"]):
                    metadata[attribute] = str(getattr(image["file"], attribute))

            if experiment == None:
                to_save_data_1.append({
                    "Nombre": image["name"],
                    "Metadata": metadata,
                    "Imagen": str(image["image"])
                })
            
            else:
                img = image["image"]
                to_save_data_2.append({
                    "Archivo": image["name"],
                    "Nombre del experimento": experiment,
                    "Tamaño de la imagen": img.size,
                    "Media": np.mean(img),
                    "Mediana": np.median(img),
                    "Moda": int(stats.mode(img.flatten(), keepdims=False).mode),
                    "Desviación estandar": np.std(img),
                })

        if len(to_save_data_1) != 0:
            collection_1.insert_many(to_save_data_1)

        if len(to_save_data_2) != 0:
            collection_2.insert_many(to_save_data_2)
