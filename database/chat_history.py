import MySQLdb
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import uuid

class ChatHistoryManager:
    """
    Quản lý lịch sử chat với session và multi-conversation
    """
    
    def __init__(self, host: str, user: str, password: str, database: str):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.init_tables()
    
    def connect_to_db(self):
        """Kết nối tới database"""
        try:
            return MySQLdb.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
        except MySQLdb.Error as e:
            print(f"Lỗi kết nối MySQL: {e}")
            return None
    
    def init_tables(self):
        """Khởi tạo các bảng cần thiết"""
        db = self.connect_to_db()
        if not db:
            return
        
        cursor = db.cursor()
        
        # Bảng users
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(100),
                role VARCHAR(20) DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        
        # Bảng chat_sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL,
                title VARCHAR(255) DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Bảng chat_messages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id VARCHAR(36) PRIMARY KEY,
                session_id VARCHAR(36) NOT NULL,
                role ENUM('user', 'assistant', 'system') NOT NULL,
                content TEXT NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_order INT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)
        
        # Bảng session_context (lưu context của session)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_context (
                id VARCHAR(36) PRIMARY KEY,
                session_id VARCHAR(36) NOT NULL,
                context_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)
        
        db.commit()
        cursor.close()
        db.close()
    
    def create_user(self, username: str, email: str = None) -> str:
        """Tạo user mới"""
        db = self.connect_to_db()
        if not db:
            return None
        
        cursor = db.cursor()
        user_id = str(uuid.uuid4())
        
        try:
            cursor.execute(
                "INSERT INTO users (id, username, email) VALUES (%s, %s, %s)",
                (user_id, username, email)
            )
            db.commit()
            return user_id
        except MySQLdb.Error as e:
            print(f"Lỗi tạo user: {e}")
            return None
        finally:
            cursor.close()
            db.close()
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Lấy user theo username"""
        db = self.connect_to_db()
        if not db:
            return None
        
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        
        try:
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            return cursor.fetchone()
        except MySQLdb.Error as e:
            print(f"Lỗi lấy user: {e}")
            return None
        finally:
            cursor.close()
            db.close()

    def delete_user(self, user_id: str) -> bool:
        """Xóa người dùng"""
        db = self.connect_to_db()
        if not db:
            return False
        
        cursor = db.cursor()
        try:
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            db.commit()
            return cursor.rowcount > 0
        except MySQLdb.Error as e:
            print(f"Lỗi xóa user: {e}")
            db.rollback()
            return False
        finally:
            cursor.close()
            db.close
    
    def create_session(self, user_id: str, title: str = "New Chat") -> str:
        """Tạo session mới"""
        db = self.connect_to_db()
        if not db:
            return None
        
        cursor = db.cursor()
        session_id = str(uuid.uuid4())
        
        try:
            cursor.execute(
                "INSERT INTO chat_sessions (id, user_id, title) VALUES (%s, %s, %s)",
                (session_id, user_id, title)
            )
            db.commit()
            return session_id
        except MySQLdb.Error as e:
            print(f"Lỗi tạo session: {e}")
            return None
        finally:
            cursor.close()
            db.close()
    
    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Lấy danh sách session của user"""
        db = self.connect_to_db()
        if not db:
            return []
        
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        
        try:
            cursor.execute("""
                SELECT s.*, 
                       (SELECT COUNT(*) FROM chat_messages WHERE session_id = s.id) as message_count,
                       (SELECT content FROM chat_messages WHERE session_id = s.id AND role = 'user' ORDER BY message_order LIMIT 1) as first_message
                FROM chat_sessions s 
                WHERE user_id = %s AND is_active = TRUE
                ORDER BY updated_at DESC 
                LIMIT %s
            """, (user_id, limit))
            return cursor.fetchall()
        except MySQLdb.Error as e:
            print(f"Lỗi lấy sessions: {e}")
            return []
        finally:
            cursor.close()
            db.close()
    
    def get_session_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        """Lấy messages của session"""
        db = self.connect_to_db()
        if not db:
            return []
        
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        
        try:
            cursor.execute("""
                SELECT * FROM chat_messages 
                WHERE session_id = %s 
                ORDER BY message_order ASC 
                LIMIT %s
            """, (session_id, limit))
            return cursor.fetchall()
        except MySQLdb.Error as e:
            print(f"Lỗi lấy messages: {e}")
            return []
        finally:
            cursor.close()
            db.close()
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None) -> str:
        """Thêm message vào session"""
        db = self.connect_to_db()
        if not db:
            return None
        
        cursor = db.cursor()
        message_id = str(uuid.uuid4())
        
        try:
            # Lấy order tiếp theo
            cursor.execute(
                "SELECT COALESCE(MAX(message_order), 0) + 1 FROM chat_messages WHERE session_id = %s",
                (session_id,)
            )
            next_order = cursor.fetchone()[0]
            
            # Thêm message
            cursor.execute("""
                INSERT INTO chat_messages (id, session_id, role, content, metadata, message_order) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (message_id, session_id, role, content, json.dumps(metadata) if metadata else None, next_order))
            
            # Cập nhật thời gian session
            cursor.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (session_id,)
            )
            
            db.commit()
            return message_id
        except MySQLdb.Error as e:
            print(f"Lỗi thêm message: {e}")
            return None
        finally:
            cursor.close()
            db.close()
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Cập nhật title session"""
        db = self.connect_to_db()
        if not db:
            return False
        
        cursor = db.cursor()
        
        try:
            cursor.execute(
                "UPDATE chat_sessions SET title = %s WHERE id = %s",
                (title, session_id)
            )
            db.commit()
            return cursor.rowcount > 0
        except MySQLdb.Error as e:
            print(f"Lỗi cập nhật title: {e}")
            return False
        finally:
            cursor.close()
            db.close()
    
    def delete_session(self, session_id: str) -> bool:
        """Xóa session (soft delete)"""
        db = self.connect_to_db()
        if not db:
            return False
        
        cursor = db.cursor()
        
        try:
            cursor.execute(
                "UPDATE chat_sessions SET is_active = FALSE WHERE id = %s",
                (session_id,)
            )
            db.commit()
            return cursor.rowcount > 0
        except MySQLdb.Error as e:
            print(f"Lỗi xóa session: {e}")
            return False
        finally:
            cursor.close()
            db.close()
    
    def get_conversation_context(self, session_id: str, last_n_messages: int = 10) -> List[Dict]:
        """Lấy context conversation cho LLM"""
        messages = self.get_session_messages(session_id, last_n_messages)
        
        context = []
        for msg in messages:
            context.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["created_at"].isoformat() if msg["created_at"] else None
            })
        
        return context
    
    def save_session_context(self, session_id: str, context_data: Dict) -> bool:
        """Lưu context của session"""
        db = self.connect_to_db()
        if not db:
            return False
        
        cursor = db.cursor()
        
        try:
            cursor.execute("SELECT id FROM chat_sessions WHERE id = %s", (session_id,))
            session_exists = cursor.fetchone()
            if not session_exists:
                return False

            cursor.execute("SELECT id FROM session_context WHERE session_id = %s", (session_id,))
            existing = cursor.fetchone()

            json_data = json.dumps(context_data, ensure_ascii=False)
            
            if existing:
                cursor.execute(
                    "UPDATE session_context SET context_data = %s, updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
                    (json_data, session_id)
                )
            else:
                context_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO session_context (id, session_id, context_data) VALUES (%s, %s, %s)",
                    (context_id, session_id, json_data)
                )
        
            db.commit()
            return True
        
        except MySQLdb.Error as e:
            print(f"Lỗi save context: {e}")
            db.rollback()
            return False
        finally:
            cursor.close()
            db.close()
    
    def get_session_context(self, session_id: str) -> Optional[Dict]:
        """Lấy context của session"""
        db = self.connect_to_db()
        if not db:
            return None
        
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        
        try:
            cursor.execute("SELECT context_data FROM session_context WHERE session_id = %s", (session_id,))
            result = cursor.fetchone()
            
            if result and result["context_data"]:
                return json.loads(result["context_data"])
            return None
        except MySQLdb.Error as e:
            print(f"Lỗi lấy context: {e}")
            return None
        finally:
            cursor.close()
            db.close()
    
    def search_messages(self, user_id: str, query: str, limit: int = 20) -> List[Dict]:
        """Tìm kiếm messages theo nội dung"""
        db = self.connect_to_db()
        if not db:
            return []
        
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        
        try:
            cursor.execute("""
                SELECT m.*, s.title as session_title
                FROM chat_messages m
                JOIN chat_sessions s ON m.session_id = s.id
                WHERE s.user_id = %s AND s.is_active = TRUE
                AND m.content LIKE %s
                ORDER BY m.created_at DESC
                LIMIT %s
            """, (user_id, f"%{query}%", limit))
            
            return cursor.fetchall()
        except MySQLdb.Error as e:
            print(f"Lỗi search messages: {e}")
            return []
        finally:
            cursor.close()
            db.close()
    
    def get_session_statistics(self, session_id: str) -> Dict:
        """Lấy thống kê session"""
        db = self.connect_to_db()
        if not db:
            return {}
        
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        
        try:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages,
                    COUNT(CASE WHEN role = 'assistant' THEN 1 END) as assistant_messages,
                    MIN(created_at) as first_message_time,
                    MAX(created_at) as last_message_time
                FROM chat_messages 
                WHERE session_id = %s
            """, (session_id,))
            
            return cursor.fetchone() or {}
        except MySQLdb.Error as e:
            print(f"Lỗi lấy thống kê: {e}")
            return {}
        finally:
            cursor.close()
            db.close()

    # --- CÁC HÀM MỚI CHO ADMIN DASHBOARD ---
    
    def get_system_stats(self) -> Dict:
        """Lấy thống kê toàn hệ thống cho Admin"""
        db = self.connect_to_db()
        if not db:
            return {}
        
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        try:
            stats = {}
            
            # Tổng user
            cursor.execute("SELECT COUNT(*) as total_users FROM users")
            stats['total_users'] = cursor.fetchone()['total_users']
            
            # Tổng session
            cursor.execute("SELECT COUNT(*) as total_sessions FROM chat_sessions")
            stats['total_sessions'] = cursor.fetchone()['total_sessions']
            
            # Session hôm nay
            cursor.execute("SELECT COUNT(*) as today_sessions FROM chat_sessions WHERE DATE(created_at) = CURDATE()")
            stats['today_sessions'] = cursor.fetchone()['today_sessions']
            
            # Thống kê số lượng điều luật (coi như documents)
            try:
                cursor.execute("SELECT COUNT(*) as total_articles FROM articles")
                stats['total_documents'] = cursor.fetchone()['total_articles']
            except:
                stats['total_documents'] = 0
            
            # Trạng thái hệ thống
            stats['system_status'] = "Online"
                
            return stats
        except MySQLdb.Error as e:
            print(f"Lỗi lấy system stats: {e}")
            return {}
        finally:
            cursor.close()
            db.close()

    def get_all_users_activity(self, limit: int = 20) -> List[Dict]:
        """Lấy danh sách user và hoạt động gần đây"""
        db = self.connect_to_db()
        if not db:
            return []
            
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        try:
            query = """
                SELECT 
                    u.id,
                    u.username,
                    MAX(s.updated_at) as last_active,
                    COUNT(s.id) as sessions
                FROM users u
                LEFT JOIN chat_sessions s ON u.id = s.user_id
                GROUP BY u.id, u.username
                ORDER BY last_active DESC
                LIMIT %s
            """
            cursor.execute(query, (limit,))
            users = cursor.fetchall()
            
            # Format datetime
            for user in users:
                if user['last_active']:
                    # Tính khoảng thời gian (ví dụ: "2 giờ trước")
                    diff = datetime.now() - user['last_active']
                    if diff.days > 0:
                        user['last_active'] = f"{diff.days} ngày trước"
                    elif diff.seconds > 3600:
                        user['last_active'] = f"{diff.seconds // 3600} giờ trước"
                    elif diff.seconds > 60:
                        user['last_active'] = f"{diff.seconds // 60} phút trước"
                    else:
                        user['last_active'] = "Vừa xong"
                else:
                    user['last_active'] = "Chưa hoạt động"
                    
            return users
        except MySQLdb.Error as e:
            print(f"Lỗi lấy user activity: {e}")
            return []
        finally:
            cursor.close()
            db.close()