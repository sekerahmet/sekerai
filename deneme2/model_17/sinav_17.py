# -*- coding: utf-8 -*-
"""sinav_17 -- TinyStories makalesinin KISA istemli sinavlari.

Makalenin 44 "Evaluation prompt"u uzun hikaye GIRISLERIDIR; modelden
yaratici bir devam ister ve GPT-4 puanlar.  Bunlar BASKA: kisa istem,
TEK dogru cevap, gozle bakilir (makale Sekil 9/10/11).

  OLGU        ortak akil bilgisi         "kediler ucar mi"
  CIKARIM     neden-sonuc, eleme         "annesi hayir dedi, o zaman KIME sordu"
  BAGLAM      kim kimdi, hangisi hangisi  "SOL cebinde ne vardi"

Makale bunlari SICAKLIK 0 ile uretti -- kiyas icin biz de oyle yapariz.
"""
from __future__ import annotations

# (istem, beklenen)  -- beklenen bir ANAHTAR, birebir eslesme degil; gozle
OLGU = [
    ('Alice was so tired when she got back home so she went',
     'to bed'),
    ('Jack and Lily saw a rainbow after a rainy day. They were amazed by '
     'the colors. Jack said, "Look, Lily. A rainbow has',
     'many colors'),
    ('Jack and Lily liked to watch the moon at night. They noticed that '
     'the moon changed its shape every night. Sometimes the moon was big '
     'and round, and sometimes it was',
     'small / thin'),
    ('Jack wanted to read a book, so he went to',
     'the library'),
    ('"Can cows fly?", Alice asked her mother.',
     'No'),
    ('"What do birds like to eat?", Tom asked his mother.',
     'worms / bugs / seeds'),
    ('"What language do they speak in France?", Tom asked his mother',
     'French'),
    ('If I throw a ball up in the air, eventually it will',
     'come down'),
    ('It was winter and cold outside so his mother told him, "You should',
     'wear a warm coat'),
]

CIKARIM = [
    ('Lily likes cats and dogs. She asked her mom for a dog and her mom '
     'said no, so instead she asked',
     'her dad'),
    ('Jack told Mary, "If you give me your banana, I\u2019ll give you my '
     'apple". Mary gave Jack her Banana so',
     'he could give her the apple'),
    ('On weekends Jack went to visit his grandmother whereas on weekdays '
     'he would go to school. Last weekend, when Jack was on his way to',
     "his grandmother's house"),
    ('Lily and Ben were having an argument. Ben said that cake is much '
     'better than ice cream and Lily said that',
     'ice cream is better'),
    ('Lily and Ben are having an argument. They are trying to decide '
     'between the park and the swimming pool. Ben says, "I want to go to '
     'the park". Lily says',
     'I want to go to the pool'),
    ('Jack\u2019s mother was not home, and his father was at home. When '
     'Jack came home, he said hello to',
     'his father'),
]

BAGLAM = [
    ('"Hi Jane, have you seen Alice? I can\u2019t find her anywhere", '
     'said Jack. Jane',
     'Alice\u2019yi aramaya yardim eder'),
    ('Max had two dogs. One was white and the other was black. Max walked '
     'up the street and saw a kid with a dog. He told the kid, "I see you '
     'have a Brown dog. I also have',
     'two dogs'),
    ('Anne had a piece of candy in her left pocket and a piece of '
     'chocolate in her right pocket. Anne\u2019s mom asked her, "Anne, '
     'what is that you have in your left pocket?"',
     'a piece of candy'),
    ('Alice had both an apple and a carrot in her bag. She took the apple '
     'out of the bag and gave it to Jack. She reached into the bag again '
     'and took',
     'out the carrot'),
    ('Alice and Jack walked up the street and met a girl in a red dress. '
     'The girl said to them, "Hi, I\u2019m Jane. What are your names?"',
     "I'm Alice and this is Jack"),
    ('Diva was hungry, and wanted to bake a cake, but she didn\u2019t have '
     'any sugar at home, so she decided to go ask around. She started '
     'walking and met a squirrel. She asked the squirrel, "Would you '
     'happen',
     'to have some sugar'),
]

SINAV = [("OLGU", OLGU), ("CIKARIM", CIKARIM), ("BAGLAM", BAGLAM)]

# Makalenin ayni istemlerdeki sonuclari -- KIYAS ZEMINI (Sekil 9/10/11).
# Hepsi sicaklik 0.  "1 katman"li model bize en yakin komsu: bizde
# katman HIC yok.
MAKALE = """
model            katman   OLGU        CIKARIM     BAGLAM
1M                  8     hicbiri     zayif       zayif
2,5M                8     kismi       kismi       kismi
8,3M                8     cogu        cogu        cogu
28M                 8     cogu        cogu        cogu
33M                 4     cogu        cogu        cogu
21M                 1     bazilari    zayif       HICBIRI
GPT2-XL           1,5B    zayif       zayif       zayif
"""
