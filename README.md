# 🎬 Film Recommener System
 
## Introduction
This film recommender system suggests movies based on the similarity of movie descriptions, which are generated using the movie's genre and tags. The system uses the CountVectorizer and TfidfTransformer from sklearn to process and transform text data into a format that can be used to calculate the cosine similarity between movies.

## 🚀 How to Start?

### Clone the repository:
```bash
git clone https://github.com/S1gge/Machine-Learning-Film-Recommender
```

### 📜 Ensure the following dependencies are installed:

- **Python 3.8** or higher

- **Pandas**

- **Numpy**

- **scikit-learn**

```bash
pip install pandas numpy scikit-learn
```

### 🗂️ Prepare the CSV files
Make sure the following files are in the correct folder:
```bash
./Film_Recommender_Systems/CSV/movies.csv
./Film_Recommender_Systems/CSV/tags.csv
```

## ▶️ Run the program.
```bash
python film_recommender.py
```

## 📢 Notes
If you get an error about missing files, check that the CSV files are in the correct location.

## 💡Conclusion.
This recommender system is a simple but effective way to suggest movies based on the similarity of their descriptions. By using text vectorization techniques like CountVectorizer and TfidfTransformer, the system processes movie genres and tags to generate useful recommendations based on cosine similarity. The program is easy to install and use and can be extended to accommodate more complex recommendation techniques or additional data sources in the future.