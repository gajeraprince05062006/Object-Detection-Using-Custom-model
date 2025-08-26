from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from db import get_db_connection
import hashlib
import threading
from model import run_system
import logging
import sys
import traceback

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Track if model is running
model_running = False

# =================================================
# Helper Functions
# =================================================
def reset_model_flag():
    """Reset the flag so the model can run again."""
    global model_running
    model_running = False


def clear_detected_items():
    """Clear all items from detected_items table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detected_items")
        conn.commit()
        cursor.close()
        conn.close()
        print("[INFO] Detected items cleared successfully")
    except Exception as e:
        print(f"[ERROR] Failed to clear detected items: {e}")


def get_detected_items_from_db():
    """Fetch detected items from database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM detected_items ORDER BY detected_at DESC")
        items = cursor.fetchall()
        cursor.close()
        conn.close()
        return items
    except Exception as e:
        print(f"[ERROR] Failed to fetch detected items: {e}")
        return []


# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


# =================================================
# Routes
# =================================================
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = hashlib.sha256(request.form['password'].encode()).hexdigest()

            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                # Clear detected items after successful login (optional)
                # Uncomment the line below if you want to clear items on login
                
                clear_detected_items()

                # Store user session
                session['user'] = user['username']
                session['fullname'] = user['fullname']
                session['role'] = user.get('role', 'user')

                flash("Login successful!", "success")
                if user.get('role') == 'admin':
                    return redirect(url_for('admin_dashboard'))
                else:
                    return redirect(url_for('dashboard'))
            else:
                flash("Invalid username or password", "danger")

        except Exception as e:
            print(f"[ERROR] Login error: {e}")
            flash("An error occurred during login. Please try again.", "danger")

    return render_template("login.html")


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            fullname = request.form['fullname']
            email = request.form['email']
            username = request.form['username']
            password = hashlib.sha256(request.form['password'].encode()).hexdigest()

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (fullname, email, username, password) VALUES (%s, %s, %s, %s)",
                (fullname, email, username, password)
            )
            conn.commit()
            cursor.close()
            conn.close()

            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))

        except Exception as e:
            print(f"[ERROR] Signup error: {e}")
            flash("Email or Username already exists.", "danger")

    return render_template("signup.html")


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('login'))
    return render_template("dashboard.html", fullname=session['fullname'])


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))


@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user' not in session or session.get('role') != 'admin':
        flash("Access denied. Admins only.", "danger")
        return redirect(url_for('login'))

    return render_template("admin_dashboard.html", fullname=session['fullname'])


@app.route('/carts')
def carts_management():
    if 'user' not in session:
        flash("Please log in to access Smart Carts.", "warning")
        return redirect(url_for('login'))
    return render_template('carts.html', fullname=session['fullname'])


@app.route('/start-shopping', methods=['GET', 'POST'])
def start_shopping():
    global model_running
    
    if 'user' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('login'))
    
    if not model_running:
        try:
            thread = threading.Thread(target=run_model_wrapper)
            thread.daemon = True
            thread.start()
            model_running = True
            flash("Smart cart started successfully.", "success")
        except Exception as e:
            print(f"[ERROR] Failed to start model: {e}")
            flash("Failed to start smart cart. Please try again.", "danger")
    else:
        flash("Smart cart is already running.", "info")

    items = get_detected_items_from_db()
    return render_template('cart.html', fullname=session.get('fullname', 'User'), items=items)


def run_model_wrapper():
    """Wrapper function to run the model and reset flag when done."""
    try:
        run_system()
    except Exception as e:
        print(f"[ERROR] Model execution failed: {e}")
    finally:
        reset_model_flag()


@app.route('/cart')
def cart_page():
    if 'user' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('login'))
    
    items = get_detected_items_from_db()
    return render_template("cart.html", fullname=session.get('fullname', 'User'), items=items)


@app.route('/all_products')
def all_products():
    if 'user' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('login'))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM products")
        products = cursor.fetchall()
        cursor.close()
        conn.close()
        return render_template('all_products.html', products=products)
    except Exception as e:
        print(f"[ERROR] Failed to fetch products: {e}")
        flash("Failed to load products.", "danger")
        return redirect(url_for('dashboard'))


@app.route('/products')
def products():
    if 'user' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('login'))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM products")
        products = cursor.fetchall()
        cursor.close()
        conn.close()
        return render_template('products.html', products=products)
    except Exception as e:
        print(f"[ERROR] Failed to fetch products: {e}")
        flash("Failed to load products.", "danger")
        return redirect(url_for('dashboard'))


@app.route('/add-to-cart/<product_name>', methods=['POST'])
def add_to_cart(product_name):
    if 'user' not in session:
        return jsonify({"message": "Please login first"}), 401
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Fetch product by name from products table
        cursor.execute("SELECT name, price FROM products WHERE name = %s", (product_name,))
        product = cursor.fetchone()

        if not product:
            cursor.close()
            conn.close()
            return jsonify({"message": "Product not found"}), 404

        # Check if product already exists in detected_items
        cursor.execute("SELECT id FROM detected_items WHERE name = %s", (product['name'],))
        existing_item = cursor.fetchone()

        if existing_item:
            cursor.close()
            conn.close()
            return jsonify({"message": f"{product['name']} is already in the cart!"})

        # Insert into detected_items if not already present
        cursor.execute("""
            INSERT INTO detected_items (name, price, detected_at)
            VALUES (%s, %s, NOW())
        """, (product['name'], product['price']))
        conn.commit()

        cursor.close()
        conn.close()
        return jsonify({"message": f"{product['name']} added to cart!"})

    except Exception as e:
        print(f"[ERROR] Failed to add product to cart: {e}")
        traceback.print_exc()
        return jsonify({"message": f"Failed to add product: {str(e)}"}), 500


@app.route('/remove-from-cart/<int:item_id>', methods=['POST'])
def remove_from_cart(item_id):
    if 'user' not in session:
        return jsonify({"message": "Please login first"}), 401
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detected_items WHERE id = %s", (item_id,))
        conn.commit()
        
        if cursor.rowcount > 0:
            cursor.close()
            conn.close()
            return jsonify({"message": "Item removed from cart!"})
        else:
            cursor.close()
            conn.close()
            return jsonify({"message": "Item not found"}), 404
            
    except Exception as e:
        print(f"[ERROR] Failed to remove item from cart: {e}")
        return jsonify({"message": f"Failed to remove item: {str(e)}"}), 500


@app.route('/clear-cart', methods=['POST'])
def clear_cart():
    if 'user' not in session:
        return jsonify({"message": "Please login first"}), 401
    
    try:
        clear_detected_items()
        return jsonify({"message": "Cart cleared successfully!"})
    except Exception as e:
        print(f"[ERROR] Failed to clear cart: {e}")
        return jsonify({"message": f"Failed to clear cart: {str(e)}"}), 500


@app.route('/payment')
def payment():
    if 'user' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('login'))
    
    amount = request.args.get('amount', '0')
    return render_template('checkout.html', amount=amount)


@app.route('/process-payment', methods=['POST'])
def process_payment():
    if 'user' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('login'))
    
    try:
        amount = request.form.get('amount')
        card_number = request.form.get('card_number')
        expiry = request.form.get('expiry')
        cvv = request.form.get('cvv')

        # Basic validation
        if not all([amount, card_number, expiry, cvv]):
            return "<h1>❌ Payment Failed: Missing required fields</h1>"

        # Simulate payment processing
        # In a real application, you would integrate with a payment gateway
        
        # Clear cart after successful payment (optional)
        clear_detected_items()
        
        return f"<h1>✅ Payment of ₹{amount} Successful!</h1><p>Thank you for your purchase!</p>"
        
    except Exception as e:
        print(f"[ERROR] Payment processing failed: {e}")
        return "<h1>❌ Payment Failed: An error occurred</h1>"


# =================================================
# Error Handlers
# =================================================
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


# =================================================
# Run App
# =================================================
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)