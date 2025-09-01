from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
#host = '192.168.1.2'
host = '192.168.3.5'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://web_user:web_password@'+host+':3306/stellar'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'users'  # 明确指定表名
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

@app.route('/')
def index():
    users = User.query.all()
    return render_template('index.html', users=users)

@app.route('/user/add', methods=['POST'])
def add_user():
    username = request.form['username']
    email = request.form['email']
    
    if User.query.filter_by(username=username).first():
        flash('用户名已存在！', 'error')
        return redirect(url_for('index'))
    
    if User.query.filter_by(email=email).first():
        flash('邮箱已存在！', 'error')
        return redirect(url_for('index'))
    
    new_user = User(username=username, email=email)
    db.session.add(new_user)
    db.session.commit()
    flash('用户添加成功！', 'success')
    return redirect(url_for('index'))

@app.route('/user/delete/<int:id>', methods=['POST'])
def delete_user(id):
    user = User.query.get_or_404(id)
    db.session.delete(user)
    db.session.commit()
    flash('用户删除成功！', 'success')
    return redirect(url_for('index'))

@app.route('/user/edit/<int:id>')
def edit_user(id):
    user = User.query.get_or_404(id)
    users = User.query.all()
    return render_template('index.html', users=users, edit_user=user)

@app.route('/user/update/<int:id>', methods=['POST'])
def update_user(id):
    user = User.query.get_or_404(id)
    username = request.form['username']
    email = request.form['email']
    
    # 检查用户名是否被其他用户使用
    existing_user = User.query.filter_by(username=username).first()
    if existing_user and existing_user.id != id:
        flash('用户名已存在！', 'error')
        return redirect(url_for('index'))
    
    # 检查邮箱是否被其他用户使用
    existing_user = User.query.filter_by(email=email).first()
    if existing_user and existing_user.id != id:
        flash('邮箱已存在！', 'error')
        return redirect(url_for('index'))
    
    user.username = username
    user.email = email
    db.session.commit()
    flash('用户更新成功！', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    #windows
    #app.run(host='0.0.0.0', port=5000, debug=False)
    #mac
    app.run(host='0.0.0.0', port=8000, debug=False) 