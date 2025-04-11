import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

class NLPClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = SVC(kernel='linear', probability=True)
        self.intents = [
            ("increase brightness", "increase_brightness"),
            ("make it brighter", "increase_brightness"),
            ("decrease brightness", "decrease_brightness"),
            ("lower the brightness", "decrease_brightness"),
            ("set brightness to 70", "set_brightness"),
            ("increase volume", "increase_volume"),
            ("raise the volume", "increase_volume"),
            ("decrease volume", "decrease_volume"),
            ("turn down the volume", "decrease_volume"),
            ("set volume to 30", "set_volume"),
            ("click photo", "click_photo"),
            ("take a picture", "click_photo"),
            ("open file manager", "open_file_manager"),
            ("show me my files", "open_file_manager"),
            ("close file manager", "close_file_manager"),
            ("open calculator", "open_calculator"),
            ("start calculator", "open_calculator"),
            ("open chrome", "open_chrome"),
            ("launch chrome browser", "open_chrome"),
            ("open vs code", "open_vscode"),
            ("start vscode", "open_vscode"),
        ]
        self.train_model()

    def train_model(self):
        texts = [text for text, label in self.intents]
        labels = [label for text, label in self.intents]
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict_intent(self, text):
        X = self.vectorizer.transform([text])
        intent = self.model.predict(X)[0]
        return intent
