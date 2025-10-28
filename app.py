"""
Flask веб-приложение для объективной оценки мужских лиц
Использует компьютерное зрение и анализ лицевых точек
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime, timedelta
import json
from face_analyzer import FaceAnalyzer
from database import Database
from flask import Flask, render_template
import os

app = Flask(__name__, template_folder='.', static_folder='.')

# Конфигурация приложения
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Создание необходимых директорий
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('database', exist_ok=True)

# Инициализация компонентов
face_analyzer = FaceAnalyzer()
db = Database('database/face_analysis.db')


def allowed_file(filename):
    """
    Проверка расширения файла
    
    Args:
        filename: имя файла
    
    Returns:
        bool: True если расширение разрешено
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def cleanup_old_files():
    """
    Удаление старых загруженных файлов (старше 1 часа)
    для защиты конфиденциальности пользователей
    """
    try:
        current_time = datetime.now()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if current_time - file_time > timedelta(hours=1):
                    os.remove(filepath)
                    print(f"Удален старый файл: {filename}")
    except Exception as e:
        print(f"Ошибка при очистке файлов: {e}")


@app.route('/')
def index():
    """Главная страница приложения"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_face():
    """
    API endpoint для анализа лица
    
    Ожидает:
        - file: изображение лица
        - user_id: опциональный ID пользователя для истории
    
    Возвращает:
        JSON с результатами анализа
    """
    # Очистка старых файлов при каждом запросе
    cleanup_old_files()
    
    # Проверка наличия файла
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не предоставлен'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Недопустимый формат файла. Используйте JPG, JPEG или PNG'}), 400
    
    try:
        # Сохранение файла с уникальным именем
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Получение user_id для истории (опционально)
        user_id = request.form.get('user_id', str(uuid.uuid4()))
        
        # Анализ лица
        analysis_result = face_analyzer.analyze_face(filepath)
        
        # Проверка успешности анализа
        if 'error' in analysis_result:
            os.remove(filepath)  # Удаление файла при ошибке
            return jsonify(analysis_result), 400
        
        # Добавление метаданных
        analysis_result['timestamp'] = datetime.now().isoformat()
        analysis_result['user_id'] = user_id
        
        # Сохранение в базу данных
        db.save_analysis(user_id, analysis_result)
        
        # Удаление файла после анализа (конфиденциальность)
        os.remove(filepath)
        
        return jsonify(analysis_result), 200
        
    except Exception as e:
        # Удаление файла при ошибке
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Ошибка при обработке: {str(e)}'}), 500


@app.route('/api/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """
    Получение истории анализов пользователя
    
    Args:
        user_id: ID пользователя
    
    Returns:
        JSON с историей анализов
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        history = db.get_user_history(user_id, limit)
        return jsonify({
            'user_id': user_id,
            'count': len(history),
            'history': history
        }), 200
    except Exception as e:
        return jsonify({'error': f'Ошибка при получении истории: {str(e)}'}), 500


@app.route('/api/stats/<user_id>', methods=['GET'])
def get_stats(user_id):
    """
    Получение статистики прогресса пользователя
    
    Args:
        user_id: ID пользователя
    
    Returns:
        JSON со статистикой
    """
    try:
        stats = db.get_user_stats(user_id)
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': f'Ошибка при получении статистики: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка работоспособности API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'analyzer_ready': face_analyzer.is_ready()
    }), 200


@app.errorhandler(413)
def too_large(e):
    """Обработка ошибки слишком большого файла"""
    return jsonify({'error': 'Файл слишком большой. Максимальный размер: 16MB'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Обработка внутренних ошибок сервера"""
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500


if __name__ == '__main__':
    print("🚀 Запуск сервера Face Analysis...")
    print("📊 Инициализация анализатора лиц...")
    
    # Проверка готовности анализатора
    if face_analyzer.is_ready():
        print("✅ Анализатор готов к работе")
    else:
        print("⚠️ Предупреждение: анализатор может работать некорректно")
    
    print("🌐 Сервер доступен по адресу: http://localhost:5000")
    print("📝 API документация:")
    print("   POST /api/analyze - анализ лица")
    print("   GET /api/history/<user_id> - история анализов")
    print("   GET /api/stats/<user_id> - статистика прогресса")
    print("   GET /api/health - проверка здоровья API")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
