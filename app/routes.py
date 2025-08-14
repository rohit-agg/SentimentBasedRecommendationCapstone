from flask import Blueprint, render_template
from . import recommendation

blueprint = Blueprint('main', __name__, template_folder='templates')

@blueprint.route('/', methods=['GET', 'POST'])
def home():
  if request.method == 'POST':
    recommendation = Recommendation()
    top_5_products = recommendation.find_top_5_products()
  return render_template('index.html')