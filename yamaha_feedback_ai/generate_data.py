"""Synthetic multilingual motorcycle complaint data generator."""
import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path
from app.utils.logger import logger
from app.utils.config import (
    RAW_DATA_DIR,
    SYNTHETIC_DATA_SIZE,
    FEEDBACK_ID_PREFIX,
    VEHICLE_MODELS,
    DOMAINS,
    FAILURES,
    COUNTRIES,
    SUPPORTED_LANGUAGES,
)

# Multilingual complaint templates
COMPLAINTS_BY_LANGUAGE = {
    "en": [
        "{failure} on my {vehicle}. Happens every time I {condition}.",
        "The {failure} is getting worse. Really frustrating.",
        "My bike has {failure}. Should not happen on a new motorcycle.",
        "Experiencing {failure}. Service center couldn't fix it.",
        "{failure} during {condition}. Very dangerous.",
        "Engine {failure} - bike won't start properly.",
        "Display keeps {failure}. Navigation stops working.",
        "Severe {failure} affecting performance.",
        "{failure} issue reported by multiple owners online.",
        "Constant {failure} since purchase.",
    ],
    "de": [
        "Mein {vehicle} hat {failure}. Passiert jedes Mal wenn ich {condition}.",
        "Das {failure} wird immer schlimmer. Sehr frustrierend.",
        "Mein Motorrad hat {failure}. Sollte nicht bei einem neuen Motorrad vorkommen.",
        "Ich habe {failure} beobachtet. Die Werkstatt konnte es nicht beheben.",
        "{failure} während {condition}. Sehr gefährlich.",
        "Motor {failure} - Motorrad startet nicht richtig.",
        "Display {failure} ständig. Navigation funktioniert nicht.",
        "Schweres {failure} beeinträchtigt die Leistung.",
        "{failure} Problem von mehreren Besitzern online gemeldet.",
        "Konstantes {failure} seit dem Kauf.",
    ],
    "da": [
        "Min {vehicle} har {failure}. Sker hver gang jeg {condition}.",
        "Det {failure} bliver værre. Meget frustrerende.",
        "Min motorcykel har {failure}. Bør ikke ske på en ny motorcykel.",
        "Jeg oplever {failure}. Servicecentret kunne ikke reparere det.",
        "{failure} under {condition}. Meget farligt.",
        "Motor {failure} - motorcykel starter ikke korrekt.",
        "Display holder ved med at {failure}. Navigation virker ikke.",
        "Alvorligt {failure} påvirker ydeevnen.",
        "{failure} problem rapporteret af flere ejere online.",
        "Konstant {failure} siden købet.",
    ],
    "pl": [
        "Mój {vehicle} ma {failure}. Dzieje się za każdym razem gdy {condition}.",
        "Problem {failure} się pogarsza. Bardzo frustrujące.",
        "Mój motocykl ma {failure}. Nie powinno się to zdarzyć na nowym motocyklu.",
        "Doświadczam {failure}. Serwis nie mógł tego naprawić.",
        "{failure} podczas {condition}. Bardzo niebezpieczne.",
        "Silnik {failure} - motocykl nie uruchamia się prawidłowo.",
        "Wyświetlacz ciągle {failure}. Nawigacja nie działa.",
        "Poważny {failure} wpływa na wydajność.",
        "Problem {failure} zgłoszony przez wielu właścicieli online.",
        "Ciągły {failure} od zakupu.",
    ],
    "fr": [
        "Ma {vehicle} a {failure}. Cela se produit chaque fois que je {condition}.",
        "Le {failure} s'aggrave. Très frustrant.",
        "Ma moto a {failure}. Ne devrait pas arriver sur une nouvelle moto.",
        "J'éprouve {failure}. Le centre de service n'a pas pu le réparer.",
        "{failure} pendant {condition}. Très dangereux.",
        "Moteur {failure} - la moto ne démarre pas correctement.",
        "L'écran continue de {failure}. La navigation ne fonctionne pas.",
        "Grave {failure} affectant les performances.",
        "Problème {failure} signalé par plusieurs propriétaires en ligne.",
        "Constant {failure} depuis l'achat.",
    ],
    "es": [
        "Mi {vehicle} tiene {failure}. Sucede cada vez que {condition}.",
        "El {failure} es cada vez peor. Muy frustrante.",
        "Mi moto tiene {failure}. No debería suceder en una moto nueva.",
        "Estoy experimentando {failure}. El centro de servicio no pudo repararlo.",
        "{failure} durante {condition}. Muy peligroso.",
        "Motor {failure} - la moto no arranca correctamente.",
        "La pantalla sigue {failure}. La navegación no funciona.",
        "Grave {failure} afectando el rendimiento.",
        "Problema {failure} reportado por múltiples propietarios en línea.",
        "Constante {failure} desde la compra.",
    ],
    "it": [
        "La mia {vehicle} ha {failure}. Accade ogni volta che {condition}.",
        "Il {failure} sta peggiorando. Molto frustrante.",
        "La mia moto ha {failure}. Non dovrebbe accadere su una moto nuova.",
        "Sto sperimentando {failure}. Il centro servizi non ha potuto ripararlo.",
        "{failure} durante {condition}. Molto pericoloso.",
        "Motore {failure} - la moto non si avvia correttamente.",
        "Il display continua a {failure}. La navigazione non funziona.",
        "Grave {failure} che influisce sulle prestazioni.",
        "Problema {failure} segnalato da più proprietari online.",
        "Costante {failure} dall'acquisto.",
    ],
    "nl": [
        "Mijn {vehicle} heeft {failure}. Gebeurt elke keer als ik {condition}.",
        "De {failure} wordt steeds erger. Erg frustrerend.",
        "Mijn motor heeft {failure}. Mag niet gebeuren op een nieuwe motor.",
        "Ik ondervind {failure}. Het servicecentrum kon het niet repareren.",
        "{failure} tijdens {condition}. Erg gevaarlijk.",
        "Motor {failure} - motor start niet correct.",
        "Display blijft {failure}. Navigatie werkt niet.",
        "Ernstige {failure} beïnvloedt de prestaties.",
        "{failure} probleem gemeld door meerdere eigenaren online.",
        "Constant {failure} sinds aankoop.",
    ],
}

DRIVING_CONDITIONS = {
    "en": ["heavy rain", "highway driving", "city traffic", "mountainous roads", "hot weather", "cold starts"],
    "de": ["starker Regen", "Autobahnfahrt", "Stadtverkehr", "bergige Straßen", "heißes Wetter", "Kaltstart"],
    "da": ["kraftig regn", "motorvejskørsel", "bytrafik", "bjergveje", "varmt vejr", "kolde startere"],
    "pl": ["intensywny deszcz", "jazda po autostradzie", "ruch miejski", "drogi górskie", "gorąca pogoda", "zimne starty"],
    "fr": ["pluie forte", "conduite autoroutière", "circulation urbaine", "routes montagneuses", "temps chaud", "démarrages à froid"],
    "es": ["lluvia fuerte", "conducción en autopista", "tráfico urbano", "caminos montañosos", "clima cálido", "arranques en frío"],
    "it": ["pioggia forte", "guida in autostrada", "traffico urbano", "strade di montagna", "tempo caldo", "avvii a freddo"],
    "nl": ["zware regen", "snelwegrijden", "stadsverkeer", "bergwegen", "heet weer", "koude start"],
}


def generate_synthetic_data(size: int = SYNTHETIC_DATA_SIZE) -> pd.DataFrame:
    """Generate synthetic multilingual motorcycle complaint data."""
    logger.info(f"Generating {size} synthetic complaint records...")
    
    records = []
    count = 0
    
    while count < size:
        language = random.choice(SUPPORTED_LANGUAGES)
        
        # Safety check: ensure templates exist for the chosen language
        if language not in COMPLAINTS_BY_LANGUAGE or language not in DRIVING_CONDITIONS:
            continue
            
        vehicle_model = random.choice(VEHICLE_MODELS)
        domain = random.choice(DOMAINS)
        failure = random.choice(FAILURES)
        country = random.choice(COUNTRIES)
        mileage = random.randint(100, 50000)
        
        # Select appropriate templates for language
        complaint_template = random.choice(COMPLAINTS_BY_LANGUAGE[language])
        condition = random.choice(DRIVING_CONDITIONS[language])
        
        # Generate feedback
        feedback = complaint_template.format(
            failure=failure,
            vehicle=vehicle_model,
            condition=condition
        )
        
        # Generate realistic timestamp (random seconds within last 90 days)
        seconds_ago = random.randint(0, 90 * 24 * 3600)
        timestamp = (datetime.now() - timedelta(seconds=seconds_ago)).isoformat()
        
        feedback_id = f"{FEEDBACK_ID_PREFIX}{count+1:06d}"
        
        records.append({
            "feedback_id": feedback_id,
            "timestamp": timestamp,
            "language": language,
            "vehicle_model": vehicle_model,
            "domain": domain,
            "customer_feedback": feedback,
            "country": country,
            "mileage": mileage,
        })
        count += 1
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} records successfully")
    return df


def save_raw_data(df: pd.DataFrame, output_path: str = None) -> str:
    """Save raw data to CSV."""
    if output_path is None:
        output_path = RAW_DATA_DIR / "yamaha_feedback.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved raw data to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    df = generate_synthetic_data()
    save_raw_data(df)
    print(df.head(10))
    print(f"\nGenerated {len(df)} records")
    print(f"Languages: {df['language'].unique()}")
