"""
Модуль для работы с базой данных
Хранение истории анализов и отслеживание прогресса
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os


class Database:
    """
    Класс для управления базой данных анализов лиц
    Использует SQLite для локального хранения
    """
    
    def __init__(self, db_path: str = 'database/face_analysis.db'):
        """
        Инициализация подключения к базе данных
        
        Args:
            db_path: путь к файлу базы данных
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Создание таблиц базы данных при первом запуске"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица анализов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                overall_rating REAL NOT NULL,
                symmetry_score REAL NOT NULL,
                proportions_score REAL NOT NULL,
                masculinity_score REAL NOT NULL,
                skin_score REAL NOT NULL,
                full_data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Индекс для быстрого поиска по user_id
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_id 
            ON analyses(user_id)
        ''')
        
        # Индекс для поиска по дате
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON analyses(timestamp)
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"✅ База данных инициализирована: {self.db_path}")
    
    def save_analysis(self, user_id: str, analysis_data: Dict) -> int:
        """
        Сохранение результатов анализа
        
        Args:
            user_id: ID пользователя
            analysis_data: данные анализа
        
        Returns:
            int: ID созданной записи
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Извлечение ключевых метрик
        overall_rating = analysis_data.get('overall_rating', 0)
        parameters = analysis_data.get('parameters', {})
        
        symmetry_score = parameters.get('symmetry', {}).get('score', 0)
        proportions_score = parameters.get('proportions', {}).get('score', 0)
        masculinity_score = parameters.get('masculinity', {}).get('score', 0)
        skin_score = parameters.get('skin_quality', {}).get('score', 0)
        
        timestamp = analysis_data.get('timestamp', datetime.now().isoformat())
        
        # Сохранение полных данных в JSON
        full_data_json = json.dumps(analysis_data, ensure_ascii=False)
        
        cursor.execute('''
            INSERT INTO analyses 
            (user_id, timestamp, overall_rating, symmetry_score, 
             proportions_score, masculinity_score, skin_score, full_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, timestamp, overall_rating, symmetry_score,
              proportions_score, masculinity_score, skin_score, full_data_json))
        
        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return analysis_id
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Получение истории анализов пользователя
        
        Args:
            user_id: ID пользователя
            limit: максимальное количество записей
        
        Returns:
            list: список анализов (от новых к старым)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, overall_rating, symmetry_score,
                   proportions_score, masculinity_score, skin_score, full_data
            FROM analyses
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'timestamp': row[1],
                'overall_rating': row[2],
                'symmetry_score': row[3],
                'proportions_score': row[4],
                'masculinity_score': row[5],
                'skin_score': row[6],
                'full_data': json.loads(row[7])
            })
        
        return history
    
    def get_user_stats(self, user_id: str) -> Dict:
        """
        Получение статистики прогресса пользователя
        
        Args:
            user_id: ID пользователя
        
        Returns:
            dict: статистика с трендами
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Получение всех анализов пользователя
        cursor.execute('''
            SELECT timestamp, overall_rating, symmetry_score,
                   proportions_score, masculinity_score, skin_score
            FROM analyses
            WHERE user_id = ?
            ORDER BY timestamp ASC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                'total_analyses': 0,
                'message': 'Нет данных для анализа'
            }
        
        # Расчет статистики
        total = len(rows)
        
        # Последний анализ
        last_analysis = {
            'timestamp': rows[-1][0],
            'overall_rating': rows[-1][1],
            'symmetry': rows[-1][2],
            'proportions': rows[-1][3],
            'masculinity': rows[-1][4],
            'skin': rows[-1][5]
        }
        
        # Средние значения
        avg_overall = sum(row[1] for row in rows) / total
        avg_symmetry = sum(row[2] for row in rows) / total
        avg_proportions = sum(row[3] for row in rows) / total
        avg_masculinity = sum(row[4] for row in rows) / total
        avg_skin = sum(row[5] for row in rows) / total
        
        # Лучший результат
        best_analysis = max(rows, key=lambda x: x[1])
        best_result = {
            'timestamp': best_analysis[0],
            'overall_rating': best_analysis[1]
        }
        
        # Прогресс (если больше одного анализа)
        progress = {}
        if total > 1:
            first_overall = rows[0][1]
            last_overall = rows[-1][1]
            progress = {
                'overall_change': round(last_overall - first_overall, 1),
                'overall_change_percent': round(((last_overall - first_overall) / first_overall * 100), 1) if first_overall > 0 else 0,
                'symmetry_change': round(rows[-1][2] - rows[0][2], 1),
                'proportions_change': round(rows[-1][3] - rows[0][3], 1),
                'masculinity_change': round(rows[-1][4] - rows[0][4], 1),
                'skin_change': round(rows[-1][5] - rows[0][5], 1)
            }
        
        return {
            'user_id': user_id,
            'total_analyses': total,
            'last_analysis': last_analysis,
            'best_result': best_result,
            'averages': {
                'overall_rating': round(avg_overall, 1),
                'symmetry': round(avg_symmetry, 1),
                'proportions': round(avg_proportions, 1),
                'masculinity': round(avg_masculinity, 1),
                'skin': round(avg_skin, 1)
            },
            'progress': progress if progress else None
        }
    
    def delete_old_analyses(self, days: int = 90):
        """
        Удаление старых анализов (для защиты конфиденциальности)
        
        Args:
            days: удалить анализы старше указанного количества дней
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        cursor.execute('''
            DELETE FROM analyses
            WHERE created_at < datetime(?, 'unixepoch')
        ''', (cutoff_date,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"🗑️ Удалено старых записей: {deleted_count}")
        return deleted_count
    
    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict]:
        """
        Получение конкретного анализа по ID
        
        Args:
            analysis_id: ID анализа
        
        Returns:
            dict или None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, user_id, timestamp, full_data
            FROM analyses
            WHERE id = ?
        ''', (analysis_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'user_id': row[1],
                'timestamp': row[2],
                'data': json.loads(row[3])
            }
        
        return None
    
    def close(self):
        """Закрытие соединения с базой данных"""
        # SQLite автоматически закрывает соединения
        pass
