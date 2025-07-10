from database import LawDocumentCreator

db = LawDocumentCreator(
    host="localhost",
    user="root",
    password="hoangducdung",
    database="law_db"
)

test = db.connect_to_db()
print(test)