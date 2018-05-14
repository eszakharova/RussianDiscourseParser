

# RussianDiscourseParser

Я хочу научиться автоматически извлекать риторические отношения между ЭДЕ на уровне предложения в письменных текстах на русском языке. 

## Подход 1 - Классификация

**Объекты** - пары ЭДЕ сгенерированные в рамках предложения и в границах окна ширины 5 (максимальное расстояние между связанными отношениями ЭДЕ внутри предложения),  [код для генерации пар ЭДЕ](https://github.com/eszakharova/RussianDiscourseParser/blob/master/prediction/preprocessing/parse_rs3_make_objects.py)

**Классы** - всего 28 классов

1. no_relation                     1866 объектов
2. joint_M                          737
3. elaboration_NS                   723
4. same-unit_M                      216
5. attribution_SN                   152
6. contrast_M                       139
7. cause-effect_SN                   82
8. cause-effect_NS                   81
9. condition_SN                      71
10. sequence_M                        66
11. elaboration_SN                    64
12. purpose_NS                        60
13. comparison_M                      47
14. attribution_NS                    44
15. background_SN                     40
16. condition_NS                      38
17. interpretation-evaluation_SN      37
18. background_NS                     35
19. concession_SN                     29
20. evidence_NS                       28
21. purpose_SN                        28
22. concession_NS                     26
23. restatement_M                     24
24. evidence_SN                       16
25. interpretation-evaluation_NS      14
26. preparation_SN                     9
27. antithesis_NS                      2
28. solutionhood_SN                    2

**Oversampling** с помощью SMOTE, в итоге в каждом классе по **1866** объектов

**Признаки:**

+ Count Vectorizer по токенам (знаки препинания учитываются, стоп-слова не удаляются, без лемматизации) - бинарные

+ Count Vectorizer по POS-тегам - бинарные

+ позиция в тексте (ЭДЕ1 и ЭДЕ2) - количественный

+ длина в токенах (ЭДЕ1 и ЭДЕ2)- количественный

+ стоит в начале предложения (ЭДЕ1 и ЭДЕ2) - бинарный

+ стоит в конце предложения (ЭДЕ1 и ЭДЕ2) - бинарный

**Качество**
+ [код сравнение разных моделей](https://github.com/eszakharova/RussianDiscourseParser/blob/master/prediction/prediction_v1_different_models.ipynb)
+ Пока что лучшая модель - Random Forest (1500 деревьев)
+ Средние метрики по всем классам на кросс-валидации по 5 фолдам:

| precision | recall | f1-score |
|-----------|--------|----------|
| 0.97      | 0.96   | 0.96     |

+ Хуже всего показатели в классе no_relation

| precision | recall | f1-score |
|-----------|--------|----------|
| 0.48      | 0.68   | 0.56     |

**Плохо распознает наличие/отстутвие отношений, но хорошо распознает типы отношений при их наличии**

**TODO:**

- [ ] Признаки - счетчик по н-граммам у POS-тегов
- [ ] Признаки - embeddings (doc2vec или word2vec)
- [ ] Признаки - синтаксическая информация
- [ ] Более точный подбор гиперпараметров (gridsearch или hyperopt)
- [ ] Дополнительная тестовая выборка (разметить вручную некоторое количество предложений)

## Подход 2. Dependecies

**TODO:**

- [ ] ???

