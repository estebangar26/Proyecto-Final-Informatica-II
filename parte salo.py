import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, segmentation
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, disk
from skimage.feature import blob_doh
import json
import os
from datetime import datetime
import sqlite3
from pymongo import MongoClient

class JPGProcessor:
    """Clase para procesar imágenes JPG y PNG"""
    
    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.processed_images = []
        
    def load_image(self, image_path):
        """Cargar imagen JPG o PNG"""
        try:
            # Leer imagen con OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Convertir de BGR a RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_image = img_rgb.copy()
            self.current_image = img_rgb.copy()
            
            return img_rgb
        except Exception as e:
            print(f"Error al cargar imagen: {e}")
            return None
    
    def change_color_space(self, image, color_space='HSV'):
        """Cambiar espacio de color de la imagen"""
        if color_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif color_space == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif color_space == 'YUV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else:
            return image
    
    def equalize_histogram(self, image):
        """Ecualización de histograma"""
        if len(image.shape) == 3:
            # Para imágenes a color
            equalized = np.zeros_like(image)
            for i in range(3):
                equalized[:, :, i] = cv2.equalizeHist(image[:, :, i])
            return equalized
        else:
            # Para imágenes en escala de grises
            return cv2.equalizeHist(image)
    
    def binarize_image(self, image, threshold_value=None):
        """Binarización de imagen"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if threshold_value is None:
            threshold_value = threshold_otsu(gray)
        
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return binary
    
    def morphological_operations(self, image, operation='close', kernel_size=5):
        """Operaciones morfológicas de cierre y apertura"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'close':
            return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        elif operation == 'close_open':
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        else:
            return gray
    
    def count_cells(self, image, min_area=50, max_area=1000):
        """Conteo de células en la imagen"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Binarización
        binary = self.binarize_image(gray)
        
        # Operaciones morfológicas para limpiar
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar por área
        valid_cells = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                valid_cells.append(contour)
        
        # Crear imagen con contornos marcados
        result_image = image.copy()
        cv2.drawContours(result_image, valid_cells, -1, (0, 255, 0), 2)
        
        return len(valid_cells), result_image, valid_cells
    
    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        """Detección de bordes con Canny (método adicional de OpenCV)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges
    
    def watershed_segmentation(self, image):
        """Segmentación por watershed (método adicional)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Aplicar threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Eliminar ruido
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Área de fondo segura
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Encontrar área de primer plano segura
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Encontrar región desconocida
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marcado de marcadores
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Añadir uno a todos los marcadores para que el fondo sea 1 en lugar de 0
        markers = markers + 1
        
        # Marcar la región desconocida con cero
        markers[unknown == 255] = 0
        
        # Aplicar watershed
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]
        
        return image


class CSVProcessor:
    """Clase para procesar datos CSV"""
    
    def __init__(self):
        self.current_data = None
        self.columns = []
        
    def load_csv(self, file_path):
        """Cargar archivo CSV"""
        try:
            self.current_data = pd.read_csv(file_path)
            self.columns = self.current_data.columns.tolist()
            return self.current_data
        except Exception as e:
            print(f"Error al cargar CSV: {e}")
            return None
    
    def get_data_info(self):
        """Obtener información del dataset"""
        if self.current_data is not None:
            return {
                'shape': self.current_data.shape,
                'columns': self.columns,
                'dtypes': self.current_data.dtypes.to_dict(),
                'null_values': self.current_data.isnull().sum().to_dict(),
                'description': self.current_data.describe().to_dict()
            }
        return None
    
    def create_scatter_plot(self, x_column, y_column, title=None):
        """Crear gráfico de dispersión entre dos columnas"""
        if self.current_data is None:
            return None
        
        if x_column not in self.columns or y_column not in self.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.current_data[x_column], self.current_data[y_column], alpha=0.7)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title or f'Gráfico de Dispersión: {x_column} vs {y_column}')
        plt.grid(True, alpha=0.3)
        return plt
    
    def get_data_table(self):
        """Retornar datos para mostrar en tabla"""
        if self.current_data is not None:
            return self.current_data
        return None
    
    def filter_data(self, column, condition, value):
        """Filtrar datos según condición"""
        if self.current_data is None:
            return None
        
        if condition == 'equal':
            return self.current_data[self.current_data[column] == value]
        elif condition == 'greater':
            return self.current_data[self.current_data[column] > value]
        elif condition == 'less':
            return self.current_data[self.current_data[column] < value]
        elif condition == 'contains':
            return self.current_data[self.current_data[column].str.contains(str(value), na=False)]
        return self.current_data
    
    def get_statistics(self, column):
        """Obtener estadísticas de una columna"""
        if self.current_data is None or column not in self.columns:
            return None
        
        if self.current_data[column].dtype in ['int64', 'float64']:
            return {
                'count': self.current_data[column].count(),
                'mean': self.current_data[column].mean(),
                'std': self.current_data[column].std(),
                'min': self.current_data[column].min(),
                'max': self.current_data[column].max(),
                'median': self.current_data[column].median(),
                'mode': self.current_data[column].mode().iloc[0] if not self.current_data[column].mode().empty else None
            }
        else:
            return {
                'count': self.current_data[column].count(),
                'unique': self.current_data[column].nunique(),
                'top': self.current_data[column].mode().iloc[0] if not self.current_data[column].mode().empty else None,
                'freq': self.current_data[column].value_counts().iloc[0] if not self.current_data[column].value_counts().empty else None
            }


class DatabaseManager:
    """Gestor de base de datos para archivos procesados"""
    
    def __init__(self, db_type='sqlite'):
        self.db_type = db_type
        if db_type == 'sqlite':
            self.setup_sqlite()
        elif db_type == 'mongo':
            self.setup_mongodb()
    
    def setup_sqlite(self):
        """Configurar base de datos SQLite"""
        self.conn = sqlite3.connect('biomedical_data.db')
        self.cursor = self.conn.cursor()
        
        # Crear tabla para archivos procesados
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                file_name TEXT NOT NULL,
                processed_date DATETIME NOT NULL,
                file_path TEXT NOT NULL,
                processing_info TEXT,
                user_id TEXT
            )
        ''')
        
        # Crear tabla para usuarios
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                user_type TEXT NOT NULL,
                created_date DATETIME NOT NULL
            )
        ''')
        
        self.conn.commit()
    
    def setup_mongodb(self):
        """Configurar base de datos MongoDB"""
        # Configurar conexión a MongoDB
        username = "seguimiento3"
        password = "mOtHxcAnhBMSNoLN"
        cluster_uri = f"mongodb+srv://{username}:{password}@info2.hfxzxrr.mongodb.net/"
        self.client = MongoClient(cluster_uri)
        self.db = self.client["biomedical_data"]
        self.processed_files = self.db["processed_files"]
        self.users = self.db["users"]
    
    def save_processed_file(self, file_id, file_type, file_name, file_path, processing_info=None, user_id=None):
        """Guardar información de archivo procesado"""
        if self.db_type == 'sqlite':
            try:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO processed_files 
                    (file_id, file_type, file_name, processed_date, file_path, processing_info, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (file_id, file_type, file_name, datetime.now(), file_path, json.dumps(processing_info), user_id))
                self.conn.commit()
                return True
            except Exception as e:
                print(f"Error al guardar en SQLite: {e}")
                return False
        
        elif self.db_type == 'mongo':
            try:
                document = {
                    'file_id': file_id,
                    'file_type': file_type,
                    'file_name': file_name,
                    'processed_date': datetime.now(),
                    'file_path': file_path,
                    'processing_info': processing_info,
                    'user_id': user_id
                }
                self.processed_files.insert_one(document)
                return True
            except Exception as e:
                print(f"Error al guardar en MongoDB: {e}")
                return False
    
    def get_processed_files(self, user_id=None):
        """Obtener archivos procesados"""
        if self.db_type == 'sqlite':
            if user_id:
                self.cursor.execute('SELECT * FROM processed_files WHERE user_id = ?', (user_id,))
            else:
                self.cursor.execute('SELECT * FROM processed_files')
            return self.cursor.fetchall()
        
        elif self.db_type == 'mongo':
            if user_id:
                return list(self.processed_files.find({'user_id': user_id}))
            else:
                return list(self.processed_files.find())


# Controlador para integrar con la arquitectura MVC
class FileProcessController:
    """Controlador para procesamiento de archivos"""
    
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.connect_signals()
    
    def connect_signals(self):
        """Conectar señales de la vista con el modelo"""
        # Conectar botones con funciones del modelo
        pass
    
    def process_jpg_file(self, file_path, user_id):
        """Procesar archivo JPG a través del modelo"""
        try:
            results = self.model.process_jpg_file(file_path, user_id)
            if results:
                self.view.display_results(results)
                return True
            return False
        except Exception as e:
            self.view.show_error(f"Error al procesar imagen: {str(e)}")
            return False
    
    def process_csv_file(self, file_path, user_id):
        """Procesar archivo CSV a través del modelo"""
        try:
            results = self.model.process_csv_file(file_path, user_id)
            if results:
                self.view.display_csv_data(results)
                return True
            return False
        except Exception as e:
            self.view.show_error(f"Error al procesar CSV: {str(e)}")
            return False
