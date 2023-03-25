from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import torch
import torch.nn.functional as F
import pandas as pd
from random import choice

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
    text_colors = ((254 ,253, 3), (0, 0, 0), (255, 255, 255))
    stroke_width = 2
    stroke_fill = 'black'

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

    def split_paragraphs(self, text, line_size=line_size):
        paragraphs = ['']
        for word in text.split(' '):
            if len(paragraphs[-1] + ' ' + word) > line_size:
                paragraphs.append(word)
            else:
                paragraphs[-1] = paragraphs[-1] + ' ' + word

        return '\n'.join(paragraphs)[1:]

    def generate_text(self):
        text = self.gptate(model.to('cpu'), tokenizer, 'токамак-странник говорит: ')
        text = text.replace('токамак-странник говорит: ', '').split('<|')[0].upper()
        try:
            text = text[text.index(' ') + 1:text.rindex('.')]
        except:
            pass
        return self.split_paragraphs(text)
        
    def get_image(self, image_text):
        text_color = choice(self.text_colors)

        search = self.unspl.search(type_='photos', per_page=50, query=image_text)
        entry = next(search.entries)
        image = Image.open(BytesIO(requests.get(entry.link_download).content))
        return entry.id, image.resize((self.image_width, round(image.size[1] / image.size[0] * self.image_width)))

    def generate(self):
        prev = pd.read_csv('bayan.csv')

        text = self.generate_text()
        image_id, image = self.get_image(text)
    
        if (text[:20] in prev['text'].unique()) or (image_id in prev['image'].unique()):
            return self.generate()
        else:
            pd.concat((
                prev, 
                pd.DataFrame({'text': [text[:20]], 'image': [image_id]}
                ))).to_csv('bayan.csv', index=False)

        image_height = image.size[1]

        draw = ImageDraw.Draw(image)
        _, _, w, h = draw.textbbox((0, 0), text, font=self.font)
        
        draw.text(((self.image_width - w) / 2, (image_height - h) / 20), text=text, 
                    fill=self.text_color, font=self.font, align='center', 
                    stroke_width=self.stroke_width, stroke_fill=self.stroke_fill)
        
        return image
