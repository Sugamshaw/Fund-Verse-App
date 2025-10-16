from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import pooling
import os

app = Flask(__name__)

# Configure MySQL connection pool
db_config = {
    "host": os.getenv("DB_HOST", "35.232.60.43"),
    "user": os.getenv("DB_USER", "fund_db"),
    "password": os.getenv("DB_PASSWORD", "Password123"),
    "database": os.getenv("DB_NAME", "fund_system"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "connection_timeout": 10
}

# ---------- CONNECTION POOL ----------
try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=5,
        **db_config
    )
except mysql.connector.Error as err:
    print(f"Error creating connection pool: {err}")
    connection_pool = None


def get_connection():
    if connection_pool:
        return connection_pool.get_connection()
    else:
        return mysql.connector.connect(**db_config)


# ---------- CLEANING UTILITY ----------
def clean_data(data):
    """Recursively clean strings in dicts, lists, or single strings."""
    if isinstance(data, list):
        return [clean_data(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    elif isinstance(data, str):
        return data.strip().replace("\r", "").replace("\n", "")
    else:
        return data


# ---------- HOME & HEALTH ----------
@app.route("/")
def home():
    return jsonify({"message": "Fintech Fund API is running!"})


@app.route("/health", methods=["GET"])
def health_check():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return jsonify({"status": "healthy", "database": "connected"}), 200
    except mysql.connector.Error as err:
        return jsonify({"status": "unhealthy", "error": str(err)}), 500


# ---------- LEGAL ENTITY ROUTES ----------
@app.route("/legal_entities", methods=["GET"])
def get_legal_entities():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM legal_entity")
        rows = clean_data(cursor.fetchall())
        cursor.close()
        conn.close()
        return jsonify(rows), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/legal_entities/<string:le_id>", methods=["GET"])
def get_legal_entity_by_id(le_id):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM legal_entity WHERE LE_ID = %s", (le_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return jsonify(clean_data(row)), 200
        return jsonify({"message": "Legal entity not found"}), 404
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/legal_entities", methods=["POST"])
def add_legal_entity():
    data = clean_data(request.get_json())
    required_fields = ["LE_ID", "LEI", "LEGAL_NAME", "JURISDICTION", "ENTITY_TYPE"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO legal_entity (LE_ID, LEI, LEGAL_NAME, JURISDICTION, ENTITY_TYPE)
            VALUES (%s, %s, %s, %s, %s)
        """, (data["LE_ID"], data["LEI"], data["LEGAL_NAME"],
              data["JURISDICTION"], data["ENTITY_TYPE"]))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Legal entity added successfully"}), 201
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


# ---------- MANAGEMENT ENTITY ROUTES ----------
@app.route("/management_entities", methods=["GET"])
def get_management_entities():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM management_entity")
        rows = clean_data(cursor.fetchall())
        cursor.close()
        conn.close()
        return jsonify(rows), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/management_entities/<string:mgmt_id>", methods=["GET"])
def get_management_entity_by_id(mgmt_id):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM management_entity WHERE MGMT_ID = %s", (mgmt_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return jsonify(clean_data(row)), 200
        return jsonify({"message": "Management entity not found"}), 404
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/management_entities", methods=["POST"])
def add_management_entity():
    data = clean_data(request.get_json())
    required_fields = ["MGMT_ID", "LE_ID", "REGISTRATION_NO", "DOMICILE", "ENTITY_TYPE"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO management_entity (MGMT_ID, LE_ID, REGISTRATION_NO, DOMICILE, ENTITY_TYPE)
            VALUES (%s, %s, %s, %s, %s)
        """, (data["MGMT_ID"], data["LE_ID"], data["REGISTRATION_NO"],
              data["DOMICILE"], data["ENTITY_TYPE"]))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Management entity added successfully"}), 201
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


# ---------- FUND MASTER ROUTES ----------
@app.route("/funds", methods=["GET"])
def get_funds():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM fund_master")
        rows = clean_data(cursor.fetchall())
        cursor.close()
        conn.close()
        return jsonify(rows), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/funds/<string:fund_id>", methods=["GET"])
def get_fund_by_id(fund_id):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM fund_master WHERE FUND_ID = %s", (fund_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return jsonify(clean_data(row)), 200
        return jsonify({"message": "Fund not found"}), 404
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/funds", methods=["POST"])
def add_fund():
    data = clean_data(request.get_json())
    required_fields = ["FUND_ID", "MGMT_ID", "LE_ID", "FUND_CODE", "FUND_NAME",
                       "FUND_TYPE", "BASE_CURRENCY", "DOMICILE", "ISIN_MASTER", "STATUS"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO fund_master (FUND_ID, MGMT_ID, LE_ID, FUND_CODE, FUND_NAME,
                                     FUND_TYPE, BASE_CURRENCY, DOMICILE, ISIN_MASTER, STATUS)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (data["FUND_ID"], data["MGMT_ID"], data["LE_ID"], data["FUND_CODE"],
              data["FUND_NAME"], data["FUND_TYPE"], data["BASE_CURRENCY"],
              data["DOMICILE"], data["ISIN_MASTER"], data["STATUS"]))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Fund added successfully"}), 201
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


# ---------- SUB FUND ROUTES ----------
@app.route("/sub_funds", methods=["GET"])
def get_sub_funds():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sub_fund")
        rows = clean_data(cursor.fetchall())
        cursor.close()
        conn.close()
        return jsonify(rows), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/sub_funds/<string:subfund_id>", methods=["GET"])
def get_sub_fund_by_id(subfund_id):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sub_fund WHERE SUBFUND_ID = %s", (subfund_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return jsonify(clean_data(row)), 200
        return jsonify({"message": "Sub fund not found"}), 404
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/sub_funds", methods=["POST"])
def add_sub_fund():
    data = clean_data(request.get_json())
    required_fields = ["SUBFUND_ID", "PARENT_FUND_ID", "LE_ID", "MGMT_ID",
                       "ISIN_SUB", "CURRENCY"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sub_fund (SUBFUND_ID, PARENT_FUND_ID, LE_ID, MGMT_ID,
                                  ISIN_SUB, CURRENCY)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (data["SUBFUND_ID"], data["PARENT_FUND_ID"], data["LE_ID"],
              data["MGMT_ID"], data["ISIN_SUB"], data["CURRENCY"]))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Sub fund added successfully"}), 201
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


# ---------- SHARE CLASS ROUTES ----------
@app.route("/share_classes", methods=["GET"])
def get_share_classes():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM share_class")
        rows = clean_data(cursor.fetchall())
        cursor.close()
        conn.close()
        return jsonify(rows), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/share_classes/<string:sc_id>", methods=["GET"])
def get_share_class_by_id(sc_id):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM share_class WHERE SC_ID = %s", (sc_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return jsonify(clean_data(row)), 200
        return jsonify({"message": "Share class not found"}), 404
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/share_classes", methods=["POST"])
def add_share_class():
    data = clean_data(request.get_json())
    required_fields = ["SC_ID", "FUND_ID", "ISIN_SC", "CURRENCY", "DISTRIBUTION",
                       "FEE_MGMT", "PERF_FEE", "EXPENSE_RATIO", "NAV", "AUM", "STATUS"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO share_class (SC_ID, FUND_ID, ISIN_SC, CURRENCY, DISTRIBUTION,
                                     FEE_MGMT, PERF_FEE, EXPENSE_RATIO, NAV, AUM, STATUS)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (data["SC_ID"], data["FUND_ID"], data["ISIN_SC"], data["CURRENCY"],
              data["DISTRIBUTION"], data["FEE_MGMT"], data["PERF_FEE"],
              data["EXPENSE_RATIO"], data["NAV"], data["AUM"], data["STATUS"]))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Share class added successfully"}), 201
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500

# ---------- LEGAL ENTITY UPDATE & DELETE ----------
@app.route("/legal_entities/<string:le_id>", methods=["PUT"])
def update_legal_entity(le_id):
    data = clean_data(request.get_json())
    required_fields = ["LEI", "LEGAL_NAME", "JURISDICTION", "ENTITY_TYPE"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE legal_entity 
            SET LEI = %s, LEGAL_NAME = %s, JURISDICTION = %s, ENTITY_TYPE = %s
            WHERE LE_ID = %s
        """, (data["LEI"], data["LEGAL_NAME"], data["JURISDICTION"], 
              data["ENTITY_TYPE"], le_id))
        
        if cursor.rowcount == 0:
            return jsonify({"message": "Legal entity not found"}), 404
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Legal entity updated successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/legal_entities/<string:le_id>", methods=["DELETE"])
def delete_legal_entity(le_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM legal_entity WHERE LE_ID = %s", (le_id,))
        
        if cursor.rowcount == 0:
            return jsonify({"message": "Legal entity not found"}), 404
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Legal entity deleted successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


# ---------- MANAGEMENT ENTITY UPDATE & DELETE ----------
@app.route("/management_entities/<string:mgmt_id>", methods=["PUT"])
def update_management_entity(mgmt_id):
    data = clean_data(request.get_json())
    required_fields = ["LE_ID", "REGISTRATION_NO", "DOMICILE", "ENTITY_TYPE"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE management_entity 
            SET LE_ID = %s, REGISTRATION_NO = %s, DOMICILE = %s, ENTITY_TYPE = %s
            WHERE MGMT_ID = %s
        """, (data["LE_ID"], data["REGISTRATION_NO"], data["DOMICILE"],
              data["ENTITY_TYPE"], mgmt_id))
        
        if cursor.rowcount == 0:
            return jsonify({"message": "Management entity not found"}), 404
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Management entity updated successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/management_entities/<string:mgmt_id>", methods=["DELETE"])
def delete_management_entity(mgmt_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM management_entity WHERE MGMT_ID = %s", (mgmt_id,))
        
        if cursor.rowcount == 0:
            return jsonify({"message": "Management entity not found"}), 404
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Management entity deleted successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


# ---------- FUND MASTER UPDATE & DELETE ----------
@app.route("/funds/<string:fund_id>", methods=["PUT"])
def update_fund(fund_id):
    data = clean_data(request.get_json())
    required_fields = ["MGMT_ID", "LE_ID", "FUND_CODE", "FUND_NAME",
                       "FUND_TYPE", "BASE_CURRENCY", "DOMICILE", "ISIN_MASTER", "STATUS"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE fund_master 
            SET MGMT_ID = %s, LE_ID = %s, FUND_CODE = %s, FUND_NAME = %s,
                FUND_TYPE = %s, BASE_CURRENCY = %s, DOMICILE = %s, 
                ISIN_MASTER = %s, STATUS = %s
            WHERE FUND_ID = %s
        """, (data["MGMT_ID"], data["LE_ID"], data["FUND_CODE"], data["FUND_NAME"],
              data["FUND_TYPE"], data["BASE_CURRENCY"], data["DOMICILE"],
              data["ISIN_MASTER"], data["STATUS"], fund_id))
        
        if cursor.rowcount == 0:
            return jsonify({"message": "Fund not found"}), 404
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Fund updated successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/funds/<string:fund_id>", methods=["DELETE"])
def delete_fund(fund_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM fund_master WHERE FUND_ID = %s", (fund_id,))
        
        if cursor.rowcount == 0:
            return jsonify({"message": "Fund not found"}), 404
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Fund deleted successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


# ---------- SUB FUND UPDATE & DELETE ----------
@app.route("/sub_funds/<string:subfund_id>", methods=["PUT"])
def update_sub_fund(subfund_id):
    data = clean_data(request.get_json())
    required_fields = ["PARENT_FUND_ID", "LE_ID", "MGMT_ID", "ISIN_SUB", "CURRENCY"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE sub_fund 
            SET PARENT_FUND_ID = %s, LE_ID = %s, MGMT_ID = %s, 
                ISIN_SUB = %s, CURRENCY = %s
            WHERE SUBFUND_ID = %s
        """, (data["PARENT_FUND_ID"], data["LE_ID"], data["MGMT_ID"],
              data["ISIN_SUB"], data["CURRENCY"], subfund_id))
        
        if cursor.rowcount == 0:
            return jsonify({"message": "Sub fund not found"}), 404
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Sub fund updated successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/sub_funds/<string:subfund_id>", methods=["DELETE"])
def delete_sub_fund(subfund_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sub_fund WHERE SUBFUND_ID = %s", (subfund_id,))
        
        if cursor.rowcount == 0:
            return jsonify({"message": "Sub fund not found"}), 404
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Sub fund deleted successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


# ---------- SHARE CLASS UPDATE & DELETE ----------
@app.route("/share_classes/<string:sc_id>", methods=["PUT"])
def update_share_class(sc_id):
    data = clean_data(request.get_json())
    required_fields = ["FUND_ID", "ISIN_SC", "CURRENCY", "DISTRIBUTION",
                       "FEE_MGMT", "PERF_FEE", "EXPENSE_RATIO", "NAV", "AUM", "STATUS"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE share_class 
            SET FUND_ID = %s, ISIN_SC = %s, CURRENCY = %s, DISTRIBUTION = %s,
                FEE_MGMT = %s, PERF_FEE = %s, EXPENSE_RATIO = %s, 
                NAV = %s, AUM = %s, STATUS = %s
            WHERE SC_ID = %s
        """, (data["FUND_ID"], data["ISIN_SC"], data["CURRENCY"], data["DISTRIBUTION"],
              data["FEE_MGMT"], data["PERF_FEE"], data["EXPENSE_RATIO"],
              data["NAV"], data["AUM"], data["STATUS"], sc_id))
        
        if cursor.rowcount == 0:
            return jsonify({"message": "Share class not found"}), 404
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Share class updated successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500


@app.route("/share_classes/<string:sc_id>", methods=["DELETE"])
def delete_share_class(sc_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM share_class WHERE SC_ID = %s", (sc_id,))
        
        if cursor.rowcount == 0:
            return jsonify({"message": "Share class not found"}), 404
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Share class deleted successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {str(err)}"}), 500
# ---------- RUN APP ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
