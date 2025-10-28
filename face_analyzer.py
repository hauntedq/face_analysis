"""
Модуль анализа лиц с использованием MediaPipe и OpenCV
Выполняет детекцию лицевых точек и расчет биометрических параметров
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import math


class FaceAnalyzer:
    """
    Класс для анализа лиц и оценки привлекательности
    Использует MediaPipe Face Mesh для детекции 468 лицевых точек
    """
    
    def __init__(self):
        """Инициализация анализатора с MediaPipe"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Индексы ключевых точек для анализа
        self.landmarks_indices = {
            'left_eye': [33, 133, 160, 159, 158, 157, 173],
            'right_eye': [362, 263, 387, 386, 385, 384, 398],
            'nose': [1, 2, 98, 327],
            'mouth': [61, 291, 0, 17, 84, 314],
            'jaw': [172, 136, 150, 176, 152, 400, 379, 365, 397, 288],
            'left_cheek': [205, 206, 207],
            'right_cheek': [425, 426, 427],
            'forehead': [10, 338, 297, 332, 284, 251],
            'chin': [152, 175, 171, 140, 194]
        }
    
    def is_ready(self) -> bool:
        """Проверка готовности анализатора"""
        return self.face_mesh is not None
    
    def analyze_face(self, image_path: str) -> Dict:
        """
        Основная функция анализа лица
        
        Args:
            image_path: путь к изображению
        
        Returns:
            dict: результаты анализа с оценками
        """
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Не удалось загрузить изображение'}
        
        # Конвертация в RGB для MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Детекция лицевых точек
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return {'error': 'Лицо не обнаружено на изображении. Убедитесь, что фото сделано анфас с хорошим освещением'}
        
        # Извлечение координат точек
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Конвертация нормализованных координат в пиксели
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append({
                'x': landmark.x * w,
                'y': landmark.y * h,
                'z': landmark.z * w  # Относительная глубина
            })
        
        # Расчет всех параметров
        symmetry_score = self._calculate_symmetry(landmarks)
        proportions_score = self._calculate_proportions(landmarks)
        masculinity_score = self._calculate_masculinity(landmarks)
        skin_score = self._analyze_skin_quality(image, landmarks)
        
        # Расчет общего рейтинга (взвешенное среднее)
        overall_rating = self._calculate_overall_rating({
            'symmetry': symmetry_score,
            'proportions': proportions_score,
            'masculinity': masculinity_score,
            'skin': skin_score
        })
        
        # Генерация рекомендаций
        recommendations = self._generate_recommendations({
            'symmetry': symmetry_score,
            'proportions': proportions_score,
            'masculinity': masculinity_score,
            'skin': skin_score
        })
        
        return {
            'overall_rating': round(overall_rating, 1),
            'parameters': {
                'symmetry': {
                    'score': round(symmetry_score, 1),
                    'description': 'Симметрия лица',
                    'interpretation': self._interpret_score(symmetry_score)
                },
                'proportions': {
                    'score': round(proportions_score, 1),
                    'description': 'Пропорции лица',
                    'interpretation': self._interpret_score(proportions_score)
                },
                'masculinity': {
                    'score': round(masculinity_score, 1),
                    'description': 'Выраженность мужских черт',
                    'interpretation': self._interpret_score(masculinity_score)
                },
                'skin_quality': {
                    'score': round(skin_score, 1),
                    'description': 'Качество кожи',
                    'interpretation': self._interpret_score(skin_score)
                }
            },
            'recommendations': recommendations,
            'face_detected': True
        }
    
    def _calculate_symmetry(self, landmarks: List[Dict]) -> float:
        """
        Расчет симметрии лица
        Сравнивает расстояния между парными точками слева и справа
        
        Args:
            landmarks: список лицевых точек
        
        Returns:
            float: оценка симметрии (0-100)
        """
        # Центральная линия лица (нос)
        nose_tip = landmarks[1]
        
        # Парные точки для сравнения
        pairs = [
            (33, 263),    # Внутренние углы глаз
            (133, 362),   # Внешние углы глаз
            (61, 291),    # Углы рта
            (234, 454),   # Скулы
            (172, 397)    # Линия челюсти
        ]
        
        asymmetry_scores = []
        
        for left_idx, right_idx in pairs:
            left_point = landmarks[left_idx]
            right_point = landmarks[right_idx]
            
            # Расстояние от центральной линии
            left_dist = abs(left_point['x'] - nose_tip['x'])
            right_dist = abs(right_point['x'] - nose_tip['x'])
            
            # Расчет процента различия
            max_dist = max(left_dist, right_dist)
            if max_dist > 0:
                diff_percent = abs(left_dist - right_dist) / max_dist
                asymmetry_scores.append(diff_percent)
        
        # Средняя асимметрия
        avg_asymmetry = np.mean(asymmetry_scores)
        
        # Конвертация в оценку 0-100 (меньше асимметрия = выше оценка)
        symmetry_score = max(0, 100 - (avg_asymmetry * 200))
        
        return symmetry_score
    
    def _calculate_proportions(self, landmarks: List[Dict]) -> float:
        """
        Расчет пропорций лица на основе золотого сечения
        и научных исследований привлекательности
        
        Args:
            landmarks: список лицевых точек
        
        Returns:
            float: оценка пропорций (0-100)
        """
        # Ключевые расстояния
        # Высота лица (от лба до подбородка)
        forehead = landmarks[10]
        chin = landmarks[152]
        face_height = self._distance(forehead, chin)
        
        # Ширина лица (между скулами)
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        face_width = self._distance(left_cheek, right_cheek)
        
        # Межглазное расстояние
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        eye_distance = self._distance(left_eye, right_eye)
        
        # Ширина носа
        nose_left = landmarks[98]
        nose_right = landmarks[327]
        nose_width = self._distance(nose_left, nose_right)
        
        # Ширина рта
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_width = self._distance(mouth_left, mouth_right)
        
        # Идеальные соотношения на основе исследований
        ideal_ratios = {
            'face_ratio': 1.618,  # Золотое сечение (высота/ширина)
            'eye_ratio': 0.46,    # Межглазное расстояние / ширина лица
            'nose_ratio': 0.25,   # Ширина носа / ширина лица
            'mouth_ratio': 0.50   # Ширина рта / ширина лица
        }
        
        # Фактические соотношения
        actual_ratios = {
            'face_ratio': face_height / face_width if face_width > 0 else 0,
            'eye_ratio': eye_distance / face_width if face_width > 0 else 0,
            'nose_ratio': nose_width / face_width if face_width > 0 else 0,
            'mouth_ratio': mouth_width / face_width if face_width > 0 else 0
        }
        
        # Расчет отклонений от идеала
        deviations = []
        for key in ideal_ratios:
            ideal = ideal_ratios[key]
            actual = actual_ratios[key]
            if ideal > 0:
                deviation = abs(ideal - actual) / ideal
                deviations.append(deviation)
        
        # Средняя точность пропорций
        avg_deviation = np.mean(deviations)
        
        # Конвертация в оценку 0-100
        proportions_score = max(0, 100 - (avg_deviation * 150))
        
        return proportions_score
    
    def _calculate_masculinity(self, landmarks: List[Dict]) -> float:
        """
        Оценка выраженности мужских черт лица
        (челюсть, подбородок, скулы, брови)
        
        Args:
            landmarks: список лицевых точек
        
        Returns:
            float: оценка мужественности (0-100)
        """
        scores = []
        
        # 1. Ширина и определенность челюсти
        jaw_left = landmarks[172]
        jaw_right = landmarks[397]
        jaw_center = landmarks[152]
        
        jaw_width = self._distance(jaw_left, jaw_right)
        
        # Ширина лица для нормализации
        cheek_left = landmarks[234]
        cheek_right = landmarks[454]
        face_width = self._distance(cheek_left, cheek_right)
        
        jaw_ratio = jaw_width / face_width if face_width > 0 else 0
        # Широкая челюсть более мужественна (идеал ~0.95)
        jaw_score = min(100, jaw_ratio * 105)
        scores.append(jaw_score)
        
        # 2. Угол челюсти (более острый = более мужественный)
        jaw_angle = self._calculate_jaw_angle(landmarks)
        # Идеальный угол ~130 градусов
        angle_score = 100 - abs(130 - jaw_angle)
        scores.append(max(0, angle_score))
        
        # 3. Высота и ширина скул
        cheekbone_prominence = self._calculate_cheekbone_prominence(landmarks)
        cheek_score = min(100, cheekbone_prominence * 100)
        scores.append(cheek_score)
        
        # 4. Выраженность подбородка
        chin_prominence = self._calculate_chin_prominence(landmarks)
        chin_score = min(100, chin_prominence * 80)
        scores.append(chin_score)
        
        # Среднее значение всех параметров
        masculinity_score = np.mean(scores)
        
        return masculinity_score
    
    def _analyze_skin_quality(self, image: np.ndarray, landmarks: List[Dict]) -> float:
        """
        Анализ качества кожи на основе текстуры и однородности
        
        Args:
            image: изображение лица
            landmarks: лицевые точки
        
        Returns:
            float: оценка качества кожи (0-100)
        """
        # Выделение области лица
        points = []
        for idx in [10, 338, 297, 332, 284, 251, 389, 356, 454, 
                    323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
                    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127]:
            if idx < len(landmarks):
                points.append([int(landmarks[idx]['x']), int(landmarks[idx]['y'])])
        
        if len(points) < 3:
            return 50.0  # Недостаточно точек
        
        # Создание маски лица
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points)], 255)
        
        # Извлечение области лица
        face_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Конвертация в оттенки серого для анализа текстуры
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # 1. Анализ однородности (стандартное отклонение яркости)
        face_pixels = gray_face[mask > 0]
        if len(face_pixels) == 0:
            return 50.0
        
        std_dev = np.std(face_pixels)
        # Меньшее отклонение = более гладкая кожа
        smoothness_score = max(0, 100 - (std_dev * 0.5))
        
        # 2. Анализ четкости краев (обнаружение дефектов)
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges[mask > 0]) / np.sum(mask > 0) if np.sum(mask) > 0 else 0
        # Меньше краев = меньше дефектов
        clarity_score = max(0, 100 - (edge_density * 2))
        
        # 3. Анализ яркости (здоровый цвет лица)
        mean_brightness = np.mean(face_pixels)
        # Оптимальная яркость 100-180
        brightness_score = 100 - abs(140 - mean_brightness) * 0.5
        brightness_score = max(0, min(100, brightness_score))
        
        # Общая оценка кожи
        skin_score = (smoothness_score * 0.4 + clarity_score * 0.4 + brightness_score * 0.2)
        
        return skin_score
    
    def _calculate_overall_rating(self, scores: Dict[str, float]) -> float:
        """
        Расчет общего рейтинга привлекательности
        
        Args:
            scores: словарь с оценками по параметрам
        
        Returns:
            float: общий рейтинг (0-100)
        """
        # Веса параметров на основе исследований привлекательности
        weights = {
            'symmetry': 0.30,      # Симметрия наиболее важна
            'proportions': 0.30,   # Пропорции также критичны
            'masculinity': 0.25,   # Мужественность черт
            'skin': 0.15          # Качество кожи
        }
        
        overall = sum(scores[key] * weights[key] for key in scores)
        
        return overall
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """
        Генерация персонализированных рекомендаций
        
        Args:
            scores: словарь с оценками
        
        Returns:
            list: список рекомендаций
        """
        recommendations = []
        
        # Рекомендации по симметрии
        if scores['symmetry'] < 70:
            recommendations.append(
                "Симметрия: Рассмотрите коррекцию прически для визуального выравнивания черт лица. "
                "Правильная укладка может компенсировать небольшую асимметрию."
            )
        
        # Рекомендации по пропорциям
        if scores['proportions'] < 70:
            recommendations.append(
                "Пропорции: Прическа и стиль бороды могут визуально улучшить пропорции лица. "
                "Консультация со стилистом поможет подобрать оптимальный образ."
            )
        
        # Рекомендации по мужественности
        if scores['masculinity'] < 70:
            recommendations.append(
                "Мужественность черт: Регулярные тренировки, особенно силовые, повышают уровень тестостерона "
                "и способствуют более выраженным мужским чертам. Правильное питание и сон также важны."
            )
        
        # Рекомендации по коже
        if scores['skin'] < 70:
            recommendations.append(
                "Качество кожи: Установите регулярный уход за кожей: очищение, увлажнение, использование SPF. "
                "Достаточный сон (7-8 часов), правильное питание и гидратация значительно улучшат состояние кожи."
            )
        elif scores['skin'] < 85:
            recommendations.append(
                "Качество кожи: Кожа в хорошем состоянии. Продолжайте регулярный уход и защиту от солнца."
            )
        
        # Общие рекомендации
        recommendations.append(
            "Общие советы: Уверенность в себе, хорошая осанка и искренняя улыбка повышают привлекательность "
            "не меньше, чем физические параметры. Работайте над внутренним состоянием."
        )
        
        recommendations.append(
            "Здоровье: Регулярные физические упражнения, сбалансированное питание, достаточный сон "
            "и управление стрессом положительно влияют на внешность."
        )
        
        return recommendations
    
    def _interpret_score(self, score: float) -> str:
        """Интерпретация оценки в текстовом виде"""
        if score >= 90:
            return "Отлично"
        elif score >= 80:
            return "Очень хорошо"
        elif score >= 70:
            return "Хорошо"
        elif score >= 60:
            return "Выше среднего"
        elif score >= 50:
            return "Среднее"
        else:
            return "Требует внимания"
    
    # Вспомогательные геометрические функции
    
    def _distance(self, point1: Dict, point2: Dict) -> float:
        """Расчет евклидова расстояния между точками"""
        return math.sqrt(
            (point1['x'] - point2['x'])**2 + 
            (point1['y'] - point2['y'])**2
        )
    
    def _calculate_jaw_angle(self, landmarks: List[Dict]) -> float:
        """Расчет угла челюсти"""
        # Точки: верх челюсти, угол челюсти, подбородок
        top_jaw = landmarks[234]
        jaw_angle_point = landmarks[172]
        chin = landmarks[152]
        
        # Векторы
        v1 = np.array([top_jaw['x'] - jaw_angle_point['x'], 
                       top_jaw['y'] - jaw_angle_point['y']])
        v2 = np.array([chin['x'] - jaw_angle_point['x'], 
                       chin['y'] - jaw_angle_point['y']])
        
        # Угол между векторами
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _calculate_cheekbone_prominence(self, landmarks: List[Dict]) -> float:
        """Расчет выраженности скул"""
        # Скулы
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        cheek_width = self._distance(left_cheek, right_cheek)
        
        # Челюсть
        jaw_left = landmarks[172]
        jaw_right = landmarks[397]
        jaw_width = self._distance(jaw_left, jaw_right)
        
        # Отношение ширины скул к ширине челюсти
        prominence = cheek_width / jaw_width if jaw_width > 0 else 0
        
        return prominence
    
    def _calculate_chin_prominence(self, landmarks: List[Dict]) -> float:
        """Расчет выраженности подбородка"""
        # Подбородок
        chin = landmarks[152]
        
        # Нижняя точка губы
        lower_lip = landmarks[17]
        
        # Точки челюсти для определения базовой линии
        jaw_left = landmarks[172]
        jaw_right = landmarks[397]
        
        # Расстояние от губы до подбородка
        chin_height = self._distance(lower_lip, chin)
        
        # Ширина челюсти для нормализации
        jaw_width = self._distance(jaw_left, jaw_right)
        
        # Относительная выраженность подбородка
        prominence = chin_height / jaw_width if jaw_width > 0 else 0
        
        return prominence