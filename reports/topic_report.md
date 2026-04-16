# Topic Modelling Report

**Dataset:** `C:\Users\manue\Desktop\mestrado\2025-2026\2Semestre\Data Mining II\hotel-review-project\data\processed\reviews_clean.csv`  
**Reviews:** 999  
**Method:** Latent Dirichlet Allocation (LDA), 6 topics  
**Vocabulary:** 1311 terms (CountVectorizer, 1–2 ngrams, min_df=3, max_df=90%)  
**LDA perplexity:** 843.86  

## Topics discovered

| # | Topic label | Reviews | Avg score | Top keywords |
| --- | --- | --- | --- | --- |
| 1 | Room Comfort & Cleanliness | 112 | 7.36 | parking, location, check, old, bed, small |
| 2 | Staff & Service | 280 | 8.62 | staff, location, friendly, clean, helpful, breakfast |
| 3 | Rooftop Bar & Ambiance | 19 | 8.47 | pillows, design, comfort, doors, extremely, like |
| 4 | Food & Breakfast | 221 | 8.31 | location, breakfast, staff, excellent, perfect, porto |
| 5 | Location & Accessibility | 360 | 8.17 | location, breakfast, staff, close, bathroom, clean |
| 6 | Check-in & WiFi | 7 | 7.43 | hear, people, dirty, charge, available, buy |

## Topic details

### 1. Room Comfort & Cleanliness

**Reviews:** 112  |  **Avg score:** 7.36

**Top keywords:** parking, location, check, old, bed, small, clean, door, reception, pay, located, booking

**Representative reviews:**

> "Perfect location, clean, and cheap. Booked a "Budget Double Room" for one night, which clearly states it has one double bed. Instead I was given a room with two (quite) small single beds. When I asked..."

> "Great location right downtown. Room and bathroom was outstanding. Bed was very comfortable. Breakfast was very good. Parking deal with hotel and parking lot was not clearly understood by both parties...."

> "The decoration is dated but fun and with character. Breakfast was fine and had everything we needed. There’s NO PARKING. My search had a filter for places with parking only, and this hotel showed as a..."

### 2. Staff & Service

**Reviews:** 280  |  **Avg score:** 8.62

**Top keywords:** staff, location, friendly, clean, helpful, breakfast, staff friendly, friendly staff, friendly helpful, comfortable, quirky, excellent

**Representative reviews:**

> "The location is excellent a few minutes to the metro about a 5 minute walk to Sao Bento station. A 10 minute walk to the river and the Riberia area. The hotel is exceptional clean. Staff friendly and ..."

> "The location is superb, close to the town hall and within 15 15-minute walk to the river front. The roof bar has a view of the town hall as well. The staff team is friendly and helpful!..."

> "Great location, very clean, friendly helpful staff Very dated. Apart from breakfast room, and a rooftop terrace there were no other facilities. No bar, no Resturant. Room was very basic and functional..."

### 3. Rooftop Bar & Ambiance

**Reviews:** 19  |  **Avg score:** 8.47

**Top keywords:** pillows, design, comfort, doors, extremely, like, wi fi, fi, wi, slept, unique, terrible

**Representative reviews:**

> "The Art Deco design and the quirky displayed items. One of the lifts kept breaking down...."

> "A unique atmosphere. Great breakfast...."

> "Internet connection rental..."

### 4. Food & Breakfast

**Reviews:** 221  |  **Avg score:** 8.31

**Top keywords:** location, breakfast, staff, excellent, perfect, porto, time, coffee, location perfect, beds, day, excellent location

**Representative reviews:**

> "We had a wonderful stay at Pão de Açúcar Hotel in Porto. From the moment we arrived, the staff were welcoming and professional, and we were pleasantly surprised with a free upgrade to a Double Room wi..."

> "The location, the staff, the quirkiness. Also our bedroom and bathroom was really big. The fridge. The rooftop outside seating The bathroom was lovely and clean but a bit weary. A bit more tabletop sp..."

> "The room was big with two double beds, fully renovated bathroom, elevator is something that is not very common for the area considering the price. Did not had breakfast so can’t comment about it. Noth..."

### 5. Location & Accessibility

**Reviews:** 360  |  **Avg score:** 8.17

**Top keywords:** location, breakfast, staff, close, bathroom, clean, old, restaurants, large, metro, like, station

**Representative reviews:**

> "Location is great, Porto is a very steep city but the hotel has a few restaurants and attractions within a walking distance. Specially the Restaurant Guarany (traditional place in Porto) only two bloc..."

> "The staff is friendly and polite. The location of the hotel is great, that is, in a walking distance to historical city center and to most important touristic attractions. Also, the metro station is v..."

> "Liked the location of the hotel Hotel felt very dated. Our room was comfortable and clean. One of the two wine bottles in the Minibar was empty and the Minibar stock list of outdated. It listed items ..."

### 6. Check-in & WiFi

**Reviews:** 7  |  **Avg score:** 7.43

**Top keywords:** hear, people, dirty, charge, available, buy, sightseeing, hallway, desk, earplugs, pillows, inside

**Representative reviews:**

> "Great shower and cool hotel in a good location. Would recommend The pillows were very soft and the mattress was very firm. The door from my room wasn’t sound proofed so unless I wore earplugs I could ..."

> "Friendliness of employees and appearance of the Hotel inside..."

> "a REALLY unique hotel! it has its own specific vibe, like no other else :) great spacious room, fair room-bar prices the only downside is the proximity of the street with regular traffic. I won't lie,..."

## Sentiment breakdown by topic

| Topic | Positive | Neutral | Negative |
| --- | --- | --- | --- |
| Check-in & WiFi | 71.4% | 14.3% | 14.3% |
| Food & Breakfast | 78.7% | 16.3% | 5.0% |
| Location & Accessibility | 77.5% | 14.2% | 8.3% |
| Rooftop Bar & Ambiance | 84.2% | 15.8% | 0.0% |
| Room Comfort & Cleanliness | 57.1% | 28.6% | 14.3% |
| Staff & Service | 86.1% | 11.8% | 2.1% |

## Figures

- `reports/figures/fig_topic_distribution.png`
- `reports/figures/fig_topic_keywords.png`
- `reports/figures/fig_topic_by_sentiment.png`
- `reports/figures/fig_topic_by_traveler.png`
