"""
Пример использования Face Analysis API
Демонстрирует работу с API через Python requests
"""

import requests
import json
from pathlib import Path


class FaceAnalysisClient:
    """Клиент для работы с Face Analysis API"""
    
    def __init__(self, base_url='http://localhost:5000'):
        """
        Инициализация клиента
        
        Args:
            base_url: базовый URL API сервера
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self):
        """
        Проверка работоспособности API
        
        Returns:
            dict: статус сервера
        """
        response = self.session.get(f'{self.base_url}/api/health')
        return response.json()
    
    def analyze_face(self, image_path, user_id=None):
        """
        Анализ лица по изображению
        
        Args:
            image_path: путь к файлу изображения
            user_id: опциональный ID пользователя
        
        Returns:
            dict: результаты анализа
        """
        # Проверка существования файла
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        
        # Подготовка данных для отправки
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {}
            
            if user_id:
                data['user_id'] = user_id
            
            # Отправка запроса
            response = self.session.post(
                f'{self.base_url}/api/analyze',
                files=files,
                data=data
            )
        
        # Обработка ответа
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json()
            raise Exception(f"Ошибка анализа: {error_data.get('error', 'Неизвестная ошибка')}")
    
    def get_history(self, user_id, limit=10):
        """
        Получение истории анализов пользователя
        
        Args:
            user_id: ID пользователя
            limit: максимальное количество записей
        
        Returns:
            dict: история анализов
        """
        response = self.session.get(
            f'{self.base_url}/api/history/{user_id}',
            params={'limit': limit}
        )
        return response.json()
    
    def get_stats(self, user_id):
        """
        Получение статистики пользователя
        
        Args:
            user_id: ID пользователя
        
        Returns:
            dict: статистика и прогресс
        """
        response = self.session.get(f'{self.base_url}/api/stats/{user_id}')
        return response.json()
    
    def print_analysis_results(self, results):
        """
        Красивый вывод результатов анализа
        
        Args:
            results: результаты анализа
        """
        print("\n" + "=" * 70)
        print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА ЛИЦА")
        print("=" * 70)
        
        # Общий рейтинг
        overall = results['overall_rating']
        print(f"\n🏆 ОБЩИЙ РЕЙТИНГ: {overall:.1f}/100")
        print(self._get_rating_bar(overall))
        
        # Детальные параметры
        print("\n📈 ДЕТАЛЬНЫЕ ПАРАМЕТРЫ:\n")
        parameters = results['parameters']
        
        for key, param in parameters.items():
            score = param['score']
            description = param['description']
            interpretation = param['interpretation']
            
            print(f"  {description}:")
            print(f"    Оценка: {score:.1f}/100 - {interpretation}")
            print(f"    {self._get_rating_bar(score, width=40)}")
            print()
        
        # Рекомендации
        if results.get('recommendations'):
            print("\n💡 РЕКОМЕНДАЦИИ:\n")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"  {i}. {rec}\n")
        
        print("=" * 70 + "\n")
    
    def _get_rating_bar(self, score, width=50):
        """
        Создание визуального бара оценки
        
        Args:
            score: оценка (0-100)
            width: ширина бара в символах
        
        Returns:
            str: визуальный бар
        """
        filled = int((score / 100) * width)
        empty = width - filled
        
        # Выбор цвета (эмуляция через символы)
        if score >= 80:
            fill_char = '█'
        elif score >= 60:
            fill_char = '▓'
        else:
            fill_char = '░'
        
        bar = fill_char * filled + '░' * empty
        return f"  [{bar}] {score:.1f}%"


def example_basic_usage():
    """Базовый пример использования клиента"""
    print("🚀 Базовый пример использования Face Analysis API\n")
    
    # Создание клиента
    client = FaceAnalysisClient('http://localhost:5000')
    
    # Проверка здоровья API
    print("1️⃣ Проверка работоспособности API...")
    try:
        health = client.check_health()
        print(f"   ✅ Статус: {health['status']}")
        print(f"   ⏰ Время: {health['timestamp']}")
        print(f"   🤖 Анализатор готов: {health['analyzer_ready']}\n")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}\n")
        return
    
    # Анализ изображения
    print("2️⃣ Анализ изображения...")
    image_path = 'test_image.jpg'  # Замените на путь к вашему изображению
    
    try:
        results = client.analyze_face(image_path, user_id='example_user_123')
        print("   ✅ Анализ завершен успешно!\n")
        
        # Вывод результатов
        client.print_analysis_results(results)
        
    except FileNotFoundError:
        print(f"   ⚠️  Файл {image_path} не найден.")
        print("   💡 Создайте тестовое изображение или укажите путь к существующему файлу.\n")
    except Exception as e:
        print(f"   ❌ Ошибка при анализе: {e}\n")


def example_history_tracking():
    """Пример отслеживания истории и прогресса"""
    print("📈 Пример отслеживания истории и прогресса\n")
    
    client = FaceAnalysisClient('http://localhost:5000')
    user_id = 'example_user_123'
    
    # Получение истории
    print("1️⃣ Получение истории анализов...")
    try:
        history_data = client.get_history(user_id, limit=5)
        
        if history_data['count'] == 0:
            print("   ℹ️  История пуста. Выполните несколько анализов для отслеживания прогресса.\n")
        else:
            print(f"   ✅ Найдено записей: {history_data['count']}\n")
            
            print("   Последние анализы:")
            for item in history_data['history'][:3]:
                timestamp = item['timestamp']
                rating = item['overall_rating']
                print(f"     • {timestamp}: Рейтинг {rating:.1f}/100")
            print()
    
    except Exception as e:
        print(f"   ❌ Ошибка: {e}\n")
    
    # Получение статистики
    print("2️⃣ Получение статистики прогресса...")
    try:
        stats = client.get_stats(user_id)
        
        if stats['total_analyses'] == 0:
            print("   ℹ️  Недостаточно данных для статистики.\n")
        else:
            print(f"   ✅ Всего анализов: {stats['total_analyses']}")
            
            # Средние значения
            averages = stats['averages']
            print(f"\n   📊 Средние значения:")
            print(f"     • Общий рейтинг: {averages['overall_rating']:.1f}")
            print(f"     • Симметрия: {averages['symmetry']:.1f}")
            print(f"     • Пропорции: {averages['proportions']:.1f}")
            print(f"     • Мужественность: {averages['masculinity']:.1f}")
            print(f"     • Качество кожи: {averages['skin']:.1f}")
            
            # Прогресс
            if stats.get('progress'):
                progress = stats['progress']
                change = progress['overall_change']
                change_percent = progress['overall_change_percent']
                
                print(f"\n   📈 Прогресс:")
                emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"     {emoji} Изменение общего рейтинга: {change:+.1f} ({change_percent:+.1f}%)")
                print(f"     • Симметрия: {progress['symmetry_change']:+.1f}")
                print(f"     • Пропорции: {progress['proportions_change']:+.1f}")
                print(f"     • Мужественность: {progress['masculinity_change']:+.1f}")
                print(f"     • Качество кожи: {progress['skin_change']:+.1f}")
            
            print()
    
    except Exception as e:
        print(f"   ❌ Ошибка: {e}\n")


def example_batch_analysis():
    """Пример пакетного анализа нескольких изображений"""
    print("📦 Пример пакетного анализа нескольких изображений\n")
    
    client = FaceAnalysisClient('http://localhost:5000')
    
    # Список изображений для анализа
    images = [
        'photo1.jpg',
        'photo2.jpg',
        'photo3.jpg'
    ]
    
    results_list = []
    
    for i, image_path in enumerate(images, 1):
        print(f"{i}️⃣ Анализ изображения: {image_path}")
        
        try:
            results = client.analyze_face(image_path, user_id='batch_user')
            results_list.append({
                'image': image_path,
                'rating': results['overall_rating'],
                'results': results
            })
            print(f"   ✅ Рейтинг: {results['overall_rating']:.1f}/100\n")
        
        except FileNotFoundError:
            print(f"   ⚠️  Файл не найден\n")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}\n")
    
    # Сравнение результатов
    if results_list:
        print("\n" + "=" * 70)
        print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 70 + "\n")
        
        # Сортировка по рейтингу
        results_list.sort(key=lambda x: x['rating'], reverse=True)
        
        for i, item in enumerate(results_list, 1):
            print(f"{i}. {item['image']}: {item['rating']:.1f}/100")
        
        print("\n" + "=" * 70 + "\n")


def example_error_handling():
    """Пример обработки ошибок"""
    print("⚠️  Пример обработки ошибок\n")
    
    client = FaceAnalysisClient('http://localhost:5000')
    
    # Тест 1: Несуществующий файл
    print("1️⃣ Попытка анализа несуществующего файла...")
    try:
        client.analyze_face('nonexistent.jpg')
    except FileNotFoundError as e:
        print(f"   ✅ Корректно обработана ошибка: {e}\n")
    
    # Тест 2: Неправильный формат (если API работает)
    print("2️⃣ Демонстрация обработки ошибок API...")
    print("   ℹ️  API должен вернуть ошибку для изображений без лица\n")
    
    # Тест 3: Недоступный сервер
    print("3️⃣ Попытка подключения к недоступному серверу...")
    offline_client = FaceAnalysisClient('http://localhost:9999')
    try:
        offline_client.check_health()
    except Exception as e:
        print(f"   ✅ Корректно обработана ошибка подключения\n")


def main():
    """Главная функция с меню примеров"""
    print("\n" + "=" * 70)
    print("🔍 FACE ANALYSIS API - ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ")
    print("=" * 70 + "\n")
    
    print("Выберите пример для запуска:\n")
    print("1. Базовое использование")
    print("2. Отслеживание истории и прогресса")
    print("3. Пакетный анализ изображений")
    print("4. Обработка ошибок")
    print("5. Запустить все примеры")
    print("0. Выход\n")
    
    choice = input("Ваш выбор (0-5): ").strip()
    
    print("\n" + "=" * 70 + "\n")
    
    if choice == '1':
        example_basic_usage()
    elif choice == '2':
        example_history_tracking()
    elif choice == '3':
        example_batch_analysis()
    elif choice == '4':
        example_error_handling()
    elif choice == '5':
        example_basic_usage()
        print("\n" + "=" * 70 + "\n")
        example_history_tracking()
        print("\n" + "=" * 70 + "\n")
        example_batch_analysis()
        print("\n" + "=" * 70 + "\n")
        example_error_handling()
    elif choice == '0':
        print("👋 До свидания!\n")
        return
    else:
        print("❌ Неверный выбор\n")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    # Простой пример использования без меню
    # Раскомментируйте для быстрого теста
    
    # client = FaceAnalysisClient()
    # results = client.analyze_face('your_photo.jpg')
    # client.print_analysis_results(results)
    
    # Или запустите интерактивное меню
    main()
