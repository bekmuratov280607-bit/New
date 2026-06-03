from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from googletrans import Translator

classes = {
    0:'World',
    1:'Sports',
    2:'Business',
    3:'Sci/Tech',
}


class CheckNews(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, 64)
    self.lstm = nn.LSTM(64, 128, batch_first=True)
    self.lin = nn.Linear(128, 4)

  def forward(self, x):
    x = self.emb(x)
    _, (x, _) = self.lstm(x)
    x = self.lin(x[-1])
    return x


vocab = torch.load("vocab (1).pth", weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CheckNews(len(vocab))
state_dict = torch.load('model (10).pth', map_location=device)

new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("fc", "lin")
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

news_app = FastAPI()

tokenizer = get_tokenizer("basic_english")

def change_audio(text):
    return [vocab[i] for i in tokenizer(text)]

class TextSchema(BaseModel):
    word: str

translator = Translator()


@news_app.post('/predict')
async def check_text(text: TextSchema):
    result = await translator.translate(text.word, dest='en')
    translate_text = result.text

    num_text = torch.tensor(change_audio(translate_text), dtype=torch.int64).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(num_text)
        result = torch.argmax(pred, dim=1).item()

        return {classes[result]}
if __name__ == "__main__":
    uvicorn.run(news_app, host="127.0.0.1", port=8000)