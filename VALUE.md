# Valeur métier mesurable

## 🎯 Problème métier ciblé
Difficulté à industrialiser les conversations vocales assistées IA (latence, cohérence, traçabilité).

## ⏱ Temps économisé (estimation)
- Hypothèse: 200 conversations/jour, 3 minutes gagnées/conversation.
- Gain estimé: **600 minutes/jour (~10h/jour)**.

## 💰 Coût évité ou réduit
- Hypothèse coût opérationnel moyen: 35 €/h.
- Économie potentielle: **~350 €/jour**, soit **~7 700 €/mois** (22 jours ouvrés).

## 🛡 Risque diminué
- Risque de non-conformité réduit grâce à la conservation des transcripts.
- Risque qualité réduit via replay/feedback et amélioration continue.

## 🚀 Capacité nouvelle créée
- Déploiement rapide d'un assistant vocal temps réel sur plusieurs canaux.
- Mesure et pilotage de la performance via métriques observabilité.

## KPIs proposés
- Latence médiane de bout en bout (ms)
- Taux d'abandon conversation (%)
- Durée moyenne de traitement (AHT)
- Taux de satisfaction feedback (thumbs up/down)
- Taux d'erreurs API realtime (%)

## Hypothèses explicites
- Volume stable à 200 conversations/jour.
- Infrastructure capable d'absorber la charge prévue.
- Qualité STT/LLM/TTS suffisante pour le domaine métier.

## Conditions de validité
- Les endpoints externes STT/LLM/TTS restent disponibles.
- Les seuils de latence restent dans les objectifs SLA.
- Les équipes exploitent effectivement les feedbacks et transcripts.
