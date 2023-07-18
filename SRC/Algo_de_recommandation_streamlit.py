import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors


popu_400_gd = pd.read_csv("SRC/popu_400.csv", sep=",")
popu_400_gd.set_index('title', inplace = True, drop = False )

image = Image.open("SRC/logo3.png")
st.image(image)
st.title('Algo-Rythme')
st.write('votre nouveau système de recommandation de musique dansante :sunglasses:')

chanson = st.text_input("Quelle est votre chanson préférée ?")
st.caption("Exemple : Girls Just Want to Have Fun, We've Got It Goin' On - Radio Edit, Whenever, Wherever")


X = popu_400_gd.select_dtypes('number').drop(['rank'], axis = 1)

# Création DF juste pour l'affichage des colonnes voulues
colounes = ['artist', 'id']
popu_a_afficher = popu_400_gd[colounes]
# popu_a_afficher.set_index('title', inplace = True)
prefixe = "https://open.spotify.com/track/"  # Préfixe à ajouter
popu_a_afficher['lien spotify'] = popu_a_afficher['id'].apply(lambda x: prefixe + x)
popu_a_afficher.drop(columns='id', inplace = True)




modelKNN = NearestNeighbors(n_neighbors = 5)

modelKNN.fit(X)
try:
    close_songs = X.loc[chanson].to_frame().T

    modelKNN.kneighbors(close_songs, n_neighbors = 5)

    neigh_dist, neigh_index = modelKNN.kneighbors(close_songs, n_neighbors = 5)

    close_songs2 = neigh_index[0][1:]
    close_songs3 = popu_a_afficher.iloc[close_songs2]


    st.write(close_songs3)
except:
    st.write('Pas de musique trouvée')
