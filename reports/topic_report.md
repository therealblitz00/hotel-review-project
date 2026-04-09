# Topic Modelling Report

**Dataset:** `C:\Users\manue\Desktop\mestrado\DAII\hotel-review-project\data\processed\reviews_clean.csv`  
**Reviews:** 999  
**Method:** Latent Dirichlet Allocation (LDA), 6 topics  
**Vocabulary:** 1289 terms (CountVectorizer, 1–2 ngrams, min_df=3, max_df=90%)  
**LDA perplexity:** 831.94  

## Topics discovered

| # | Topic label | Reviews | Avg score | Top keywords |
| --- | --- | --- | --- | --- |
| 1 | Check-in & WiFi | 86 | 7.63 | location, key, station, reception, wifi, time |
| 2 | Facilities & Comfort | 95 | 8.13 | uncomfortable, air conditioning, conditioning, air, era, algo |
| 3 | Room Comfort & Cleanliness | 290 | 8.0 | location, staff, breakfast, clean, bed, friendly |
| 4 | Location & Convenience | 92 | 8.53 | location, breakfast, porto, perfect, bar, open |
| 5 | Staff & Service | 341 | 8.52 | location, staff, breakfast, friendly, helpful, clean |
| 6 | Value & Overall Experience | 95 | 8.35 | location, walking, recommend, distance, staff, walking distance |

## Topic details

### 1. Check-in & WiFi

**Reviews:** 86  |  **Avg score:** 7.63

**Top keywords:** location, key, station, reception, wifi, time, leave, like, metro, metro station, door, check

**Representative reviews:**

> "24 hours English-speaking reception, next to Subway/Metro, in the heart of the old city, walkable to most attractions. The front door was locked late at night for better security. Iron board and iron ..."

> "Great location within walking distance of everything you are likely to want to see. Lots of decent bars and restaurants nearby. Very pleasant staff. Wifi signal in the bedroom was weak. I suggest to a..."

> "Third time in this hotel. Beside nice and clean rooms, the location is great. 250 m from Aliados metro station, and 1 station from Trindade station who connect the airport to city center and cost 2.75..."

### 2. Facilities & Comfort

**Reviews:** 95  |  **Avg score:** 8.13

**Top keywords:** uncomfortable, air conditioning, conditioning, air, era, algo, ma, emplacement, cama, hôtel, centrale, na

**Representative reviews:**

> "Está céntrico, el desayuno muy variado y organizado, el personal de recepción muy educado, habitación grande (éramos matrimonio y niño) aunque nuestra cama algo justa Podía estar algo más limpio. Habí..."

> "Personnel très agréable , gentil et professionnel L'emplacement de l'établissement était parfait La décoration de l'ensemble est très agréable La taille de la chambre et la propreté de l'hôtel Un brui..."

> "Easy to get to. Near two metro stations. Plenty of eating options nearby. Nothing...."

### 3. Room Comfort & Cleanliness

**Reviews:** 290  |  **Avg score:** 8.0

**Top keywords:** location, staff, breakfast, clean, bed, friendly, close, small, helpful, walk, bathroom, coffee

**Representative reviews:**

> "Good location. Quirky decor throughout hotel. Clean rooms with decent facilities. Small fridge and digital safe. Small reception area. We arrived 10 minutes before check in time, and although we were ..."

> "Very unique hotel with vintage 1950s items for decor. The staff was incredibly friendly and very helpful. The breakfasts were good (I am a big fan of starting the day with pastel de nata) although the..."

> "The staff were very helpful and informative. I arrived in the morning so I left my bag at the hotel while I toured the city. When I came to check in they had already put my bag in my room. The room it..."

### 4. Location & Convenience

**Reviews:** 92  |  **Avg score:** 8.53

**Top keywords:** location, breakfast, porto, perfect, bar, open, price, staff, clean, city, convenient, located

**Representative reviews:**

> "Spacious room and balcony with good facilities. We found a lot to like for the breakfast, especially the fresh fruit, cheese, breads, cakes, and cereals. Quirky decor. Great welcome from the check in ..."

> "I liked the staff very clear and receptive. Bed was big. Very clean. Right in the heart of the city. Sorted our luggage. Had lots of TV channels. Window shutters were permanently closed. Had snacks ou..."

> "Perfect location, room was big and clean, comfortable bed, good breakfast..."

### 5. Staff & Service

**Reviews:** 341  |  **Avg score:** 8.52

**Top keywords:** location, staff, breakfast, friendly, helpful, clean, old, excellent, friendly staff, like, quirky, location staff

**Representative reviews:**

> "The hotel contains a fabulous collection of mid-century artefacts, including dodgems, phones, typewriters and assorted pieces of funky, kitsch or otherwise rather stylish furniture. If you like this k..."

> "The vintage style and vibe. All the collected retro toys and bumper cars - this place definitely has a unique climate. They even have proper regular room keys with huge tags. It feels like being trans..."

> "The location, the staff, the quirkiness. Also our bedroom and bathroom was really big. The fridge. The rooftop outside seating The bathroom was lovely and clean but a bit weary. A bit more tabletop sp..."

### 6. Value & Overall Experience

**Reviews:** 95  |  **Avg score:** 8.35

**Top keywords:** location, walking, recommend, distance, staff, walking distance, porto, money, breakfast, value, ok, value money

**Representative reviews:**

> "Location is great, Porto is a very steep city but the hotel has a few restaurants and attractions within a walking distance. Specially the Restaurant Guarany (traditional place in Porto) only two bloc..."

> "Excellent location allowing us to explore the city by foot. Some extraordinary restaurants are reachable within few minutes: Churrasqueira Moura and Lagostim. This hotel is for travelers who enjoy bei..."

> "Location central We stayed at this hotel expecting a comfortable night, especially for €190 per night including breakfast, but unfortunately it didn’t quite meet expectations. The main issue was the m..."

## Sentiment breakdown by topic

| Topic | Positive | Neutral | Negative |
| --- | --- | --- | --- |
| Check-in & WiFi | 70.9% | 15.1% | 14.0% |
| Facilities & Comfort | 83.2% | 12.6% | 4.2% |
| Location & Convenience | 83.7% | 13.0% | 3.3% |
| Room Comfort & Cleanliness | 71.0% | 20.7% | 8.3% |
| Staff & Service | 83.9% | 12.0% | 4.1% |
| Value & Overall Experience | 73.7% | 18.9% | 7.4% |

## Figures

- `reports/figures/fig_topic_distribution.png`
- `reports/figures/fig_topic_keywords.png`
- `reports/figures/fig_topic_by_sentiment.png`
- `reports/figures/fig_topic_by_traveler.png`
