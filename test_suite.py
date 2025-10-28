"""
Набор тестов для приложения Face Analysis
Тестирование основных функций анализа и работы с БД
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
from datetime import datetime
import numpy as np

# Добавление родительской директории в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import Database


class TestDatabase(unittest.TestCase):
    """Тесты для модуля базы данных"""
    
    def setUp(self):
        """Создание временной БД перед каждым тестом"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.db = Database(self.db_path)
        
    def tearDown(self):
        """Очистка после каждого теста"""
        shutil.rmtree(self.temp_dir)
    
    def test_database_creation(self):
        """Тест создания базы данных"""
        self.assertTrue(os.path.exists(self.db_path))
        
    def test_save_analysis(self):
        """Тест сохранения анализа"""
        test_data = {
            'overall_rating': 75.5,
            'parameters': {
                'symmetry': {'score': 80.0},
                'proportions': {'score': 75.0},
                'masculinity': {'score': 70.0},
                'skin_quality': {'score': 77.0}
            },
            'recommendations': ['Test recommendation'],
            'timestamp': datetime.now().isoformat()
        }
        
        analysis_id = self.db.save_analysis('test_user', test_data)
        self.assertIsNotNone(analysis_id)
        self.assertGreater(analysis_id, 0)
    
    def test_get_user_history(self):
        """Тест получения истории пользователя"""
        # Сохранение нескольких анализов
        for i in range(3):
            test_data = {
                'overall_rating': 70.0 + i,
                'parameters': {
                    'symmetry': {'score': 75.0},
                    'proportions': {'score': 70.0},
                    'masculinity': {'score': 65.0},
                    'skin_quality': {'score': 72.0}
                },
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            }
            self.db.save_analysis('test_user', test_data)
        
        # Получение истории
        history = self.db.get_user_history('test_user', limit=10)
        self.assertEqual(len(history), 3)
        
        # Проверка сортировки (от новых к старым)
        self.assertGreaterEqual(history[0]['overall_rating'], history[-1]['overall_rating'])
    
    def test_get_user_stats(self):
        """Тест получения статистики пользователя"""
        # Сохранение анализов с разными оценками
        ratings = [70.0, 75.0, 80.0]
        for rating in ratings:
            test_data = {
                'overall_rating': rating,
                'parameters': {
                    'symmetry': {'score': rating},
                    'proportions': {'score': rating},
                    'masculinity': {'score': rating},
                    'skin_quality': {'score': rating}
                },
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            }
            self.db.save_analysis('test_user', test_data)
        
        # Получение статистики
        stats = self.db.get_user_stats('test_user')
        
        self.assertEqual(stats['total_analyses'], 3)
        self.assertIn('last_analysis', stats)
        self.assertIn('best_result', stats)
        self.assertIn('averages', stats)
        self.assertIn('progress', stats)
        
        # Проверка прогресса
        self.assertEqual(stats['progress']['overall_change'], 10.0)
    
    def test_empty_user_history(self):
        """Тест для пользователя без истории"""
        history = self.db.get_user_history('nonexistent_user', limit=10)
        self.assertEqual(len(history), 0)
    
    def test_empty_user_stats(self):
        """Тест статистики для пользователя без данных"""
        stats = self.db.get_user_stats('nonexistent_user')
        self.assertEqual(stats['total_analyses'], 0)


class TestFaceAnalyzerGeometry(unittest.TestCase):
    """Тесты геометрических функций анализатора"""
    
    def setUp(self):
        """Импорт анализатора если возможно"""
        try:
            from face_analyzer import FaceAnalyzer
            self.analyzer = FaceAnalyzer()
            self.analyzer_available = True
        except ImportError:
            self.analyzer_available = False
            print("⚠️ MediaPipe не установлен, пропуск тестов анализатора")
    
    def test_distance_calculation(self):
        """Тест расчета расстояния между точками"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        point1 = {'x': 0, 'y': 0, 'z': 0}
        point2 = {'x': 3, 'y': 4, 'z': 0}
        
        distance = self.analyzer._distance(point1, point2)
        self.assertAlmostEqual(distance, 5.0, places=2)
    
    def test_symmetry_perfect(self):
        """Тест расчета симметрии для идеально симметричного лица"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        # Создание симметричных тестовых точек
        landmarks = []
        for i in range(500):
            landmarks.append({'x': 100, 'y': 100, 'z': 0})
        
        # Симметричные точки относительно центра
        landmarks[33] = {'x': 80, 'y': 100, 'z': 0}   # левый глаз
        landmarks[263] = {'x': 120, 'y': 100, 'z': 0}  # правый глаз
        landmarks[1] = {'x': 100, 'y': 100, 'z': 0}    # нос (центр)
        
        score = self.analyzer._calculate_symmetry(landmarks)
        
        # Идеальная симметрия должна давать высокую оценку
        self.assertGreater(score, 90)
    
    def test_interpret_score(self):
        """Тест интерпретации оценок"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        self.assertEqual(self.analyzer._interpret_score(95), "Отлично")
        self.assertEqual(self.analyzer._interpret_score(85), "Очень хорошо")
        self.assertEqual(self.analyzer._interpret_score(75), "Хорошо")
        self.assertEqual(self.analyzer._interpret_score(65), "Выше среднего")
        self.assertEqual(self.analyzer._interpret_score(55), "Среднее")
        self.assertEqual(self.analyzer._interpret_score(45), "Требует внимания")


class TestRecommendations(unittest.TestCase):
    """Тесты генерации рекомендаций"""
    
    def setUp(self):
        """Инициализация анализатора"""
        try:
            from face_analyzer import FaceAnalyzer
            self.analyzer = FaceAnalyzer()
            self.analyzer_available = True
        except ImportError:
            self.analyzer_available = False
    
    def test_low_symmetry_recommendation(self):
        """Тест рекомендации при низкой симметрии"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        scores = {
            'symmetry': 60,
            'proportions': 80,
            'masculinity': 75,
            'skin': 80
        }
        
        recommendations = self.analyzer._generate_recommendations(scores)
        
        # Должна быть рекомендация по симметрии
        symmetry_rec = any('симметр' in rec.lower() for rec in recommendations)
        self.assertTrue(symmetry_rec)
    
    def test_low_skin_recommendation(self):
        """Тест рекомендации при низком качестве кожи"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        scores = {
            'symmetry': 80,
            'proportions': 80,
            'masculinity': 75,
            'skin': 60
        }
        
        recommendations = self.analyzer._generate_recommendations(scores)
        
        # Должна быть рекомендация по коже
        skin_rec = any('кож' in rec.lower() for rec in recommendations)
        self.assertTrue(skin_rec)
    
    def test_all_good_scores(self):
        """Тест рекомендаций при хороших оценках"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        scores = {
            'symmetry': 85,
            'proportions': 85,
            'masculinity': 85,
            'skin': 85
        }
        
        recommendations = self.analyzer._generate_recommendations(scores)
        
        # Должны быть общие рекомендации
        self.assertGreater(len(recommendations), 0)


class TestOverallRating(unittest.TestCase):
    """Тесты расчета общего рейтинга"""
    
    def setUp(self):
        """Инициализация анализатора"""
        try:
            from face_analyzer import FaceAnalyzer
            self.analyzer = FaceAnalyzer()
            self.analyzer_available = True
        except ImportError:
            self.analyzer_available = False
    
    def test_perfect_scores(self):
        """Тест с идеальными оценками"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        scores = {
            'symmetry': 100,
            'proportions': 100,
            'masculinity': 100,
            'skin': 100
        }
        
        overall = self.analyzer._calculate_overall_rating(scores)
        self.assertEqual(overall, 100.0)
    
    def test_average_scores(self):
        """Тест со средними оценками"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        scores = {
            'symmetry': 70,
            'proportions': 70,
            'masculinity': 70,
            'skin': 70
        }
        
        overall = self.analyzer._calculate_overall_rating(scores)
        self.assertEqual(overall, 70.0)
    
    def test_weighted_average(self):
        """Тест взвешенного среднего"""
        if not self.analyzer_available:
            self.skipTest("Analyzer not available")
        
        scores = {
            'symmetry': 100,  # вес 0.30
            'proportions': 100,  # вес 0.30
            'masculinity': 50,  # вес 0.25
            'skin': 50  # вес 0.15
        }
        
        overall = self.analyzer._calculate_overall_rating(scores)
        
        # Ожидаемое значение: 100*0.3 + 100*0.3 + 50*0.25 + 50*0.15 = 80
        self.assertAlmostEqual(overall, 80.0, places=1)


class TestAnalyzerReadiness(unittest.TestCase):
    """Тесты готовности анализатора"""
    
    def test_analyzer_initialization(self):
        """Тест инициализации анализатора"""
        try:
            from face_analyzer import FaceAnalyzer
            analyzer = FaceAnalyzer()
            self.assertTrue(analyzer.is_ready())
        except ImportError:
            self.skipTest("MediaPipe not installed")


def run_tests():
    """Запуск всех тестов с подробным выводом"""
    print("=" * 70)
    print("🧪 ЗАПУСК ТЕСТОВ FACE ANALYSIS")
    print("=" * 70)
    print()
    
    # Создание тестового набора
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавление всех тестовых классов
    suite.addTests(loader.loadTestsFromTestCase(TestDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestFaceAnalyzerGeometry))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendations))
    suite.addTests(loader.loadTestsFromTestCase(TestOverallRating))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyzerReadiness))
    
    # Запуск тестов с подробным выводом
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print(f"✅ Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Провалено: {len(result.failures)}")
    print(f"⚠️  Ошибок: {len(result.errors)}")
    print(f"⏭️  Пропущено: {len(result.skipped)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
