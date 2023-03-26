from transformers import AutoTokenizer, AutoModelForCausalLM
from numpy.random import choice
import torch
import torch.nn.functional as F
import pandas as pd

from io import BytesIO
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import requests
from pyunsplash import PyUnsplash

tokenizer = AutoTokenizer.from_pretrained('tokenizer')
model = AutoModelForCausalLM.from_pretrained('model')        

class Generator():
    api_key = open('unsplash_token.txt').read().split()[0]
    unspl = PyUnsplash(api_key=api_key)
    
    image_width = 1024
    font_size = 60
    line_size = round(image_width / font_size * 1.9)
    
    font = ImageFont.truetype('impact.ttf', size=font_size)
    text_colors = {'yellow': (254 ,253, 3), 'black': (0, 0, 0), 'white': (255, 255, 255)}
    text_color_freq = (0.75, 0.2, 0.05)
    stroke_width = 2
    stroke_fill = 'black'

    # гпт-генератор
    def gptate(
            self,
            model,
            tokenizer,
            prompt,
            entry_length=50, #maximum number of words
            top_p=0.8,
            temperature=1.,
        ):
        
        model.eval()
        generated_num = 0
        generated_list = []

        filter_value = -float("Inf")

        with torch.no_grad():

            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                if next_token == tokenizer.encode('<|endoftext|>'):
                    break

                generated = torch.cat((generated, next_token), dim=1)

        output_list = list(generated.squeeze().numpy())
        output_text = tokenizer.decode(output_list) 
                    
        return output_text

    # добавление разделителей строк, чтобы не залезали за край
    def split_paragraphs(self, text, line_size=line_size):
        paragraphs = ['']
        for word in text.split(' '):
            if len(paragraphs[-1] + ' ' + word) > line_size:
                paragraphs.append(word)
            else:
                paragraphs[-1] = paragraphs[-1] + ' ' + word

        return '\n'.join(paragraphs)[1:]

    # генерация текста вообще
    def generate_text(self):
        text = self.gptate(model.to('cpu'), tokenizer, ' ').upper()
        try:
            # обрезаем до последнего законченного
            text = text['''text.index(' ') + 1''':text.rindex('.')]
        except:
            pass
        return self.split_paragraphs(text)
        
    # достаем картинку по запросу
    def get_image(self, image_text, prev):
        search = self.unspl.search(type_='photos', per_page=50, query=image_text)
        # мы не хотим, чтобы картинки повторялись
        for entry in search.entries:
            if entry.id not in prev.unique():
                image = Image.open(BytesIO(requests.get(entry.link_download).content))
                return image.resize((self.image_width, round(image.size[1] / image.size[0] * self.image_width))), entry.id
        return

    # генерируем
    def generate(self):
        # мы не хотим, чтобы тексты повторялись
        prev = pd.read_csv('bayan.csv')

        print('generating text')
        while True:
            text = self.generate_text()
            if text[:20] not in prev['text'].unique():
                break

        print('generating image')
        image, image_id = self.get_image(text, prev['image'])
        if not image:
            self.generate()

        # добавляем текст и картинку в список баянов    
        pd.concat((
            prev, 
            pd.DataFrame({'text': [text[:20]], 'image': [image_id]}
                        ))).to_csv('bayan.csv', index=False)

        # рисуем
        image_height = image.size[1]

        draw = ImageDraw.Draw(image)
        _, _, w, h = draw.textbbox((0, 0), text, font=self.font)
        
        text_color = self.text_colors[choice(list(self.text_colors.keys()), 1, p=self.text_color_freq)[0]]
        draw.text(((self.image_width - w) / 2, (image_height - h) / 20), text=text, 
                    fill=text_color, font=self.font, align='center', 
                    stroke_width=self.stroke_width, stroke_fill=self.stroke_fill)
        
        return image
