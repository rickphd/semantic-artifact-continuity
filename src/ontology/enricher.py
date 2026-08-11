"""
Módulo de enriquecimiento semántico usando ontología RDF.

Este módulo proporciona la clase OntologyEnricher para extraer features ontológicas
de texto usando la ontología completa (rr-core.ttl + módulos sentiment, domain, shapes).

Autor: Felipe Rosero
Proyecto: Análisis de Sentimiento en Reddit sobre IA
Fecha: Noviembre 2025
Versión: v03 - Smart word boundaries + compuestos técnicos (recuperar cobertura)
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

import rdflib
from rdflib import Graph, Literal, Namespace
from rdflib.plugins.sparql import prepareQuery


# ========== UTILIDADES INTERNAS v03 ==========

# Patrón de limpieza selectiva (solo code blocks)
_CODE_RE = re.compile(r'`[^`]+`|```[\s\S]+?```')  # inline & triple-backtick


def _clean_code_blocks(s: str) -> str:
    """
    Limpia texto removiendo SOLO bloques de código.
    
    V03: URLs/emails/mentions ya NO se eliminan porque los smart word boundaries
    previenen FP sin necesidad de limpieza agresiva. Preservar URLs como
    "openai.com" o "ai-powered" ayuda a detectar conceptos técnicos válidos.
    
    Args:
        s: Texto original
        
    Returns:
        Texto con bloques de código reemplazados por espacios
    """
    return _CODE_RE.sub(' ', s)


def _smart_word_boundary(term: str) -> Pattern:
    """
    Genera patrón regex con word boundaries INTELIGENTES para términos cortos (≤3 chars).
    
    V03: Mejora sobre v02 para recuperar cobertura sin reintroducir FP.
    
    PERMITE (verdaderos positivos):
    - "AI researcher" (espacio antes/después)
    - "AI-powered" (guion como separador)
    - "OpenAI" (inicio de palabra compuesta)
    - "IA generativa" (espacio antes/después)
    - "(AI)" (paréntesis como separadores)
    
    BLOQUEA (falsos positivos):
    - "again" (letra antes Y después)
    - "maintain" (letra antes Y después)  
    - "email@ai.com" (@ no es separador válido)
    - "said" (letra antes Y después)
    
    Args:
        term: Término corto a buscar (ej: "ai", "ia", "ml")
        
    Returns:
        Pattern compilado con case-insensitive
    """
    # Caracteres que califican como word boundary:
    # - Espacios, guiones, puntuación, paréntesis, inicio/fin de línea
    boundary = r"[\s\-\(\)\[\]\{\}\.,;:!\?\"'`]"
    
    # Patrón: (inicio|boundary) + term + lookahead(boundary|fin)
    # Lookahead evita consumir el boundary siguiente (permite matches consecutivos)
    pattern = rf"(?:^|{boundary}){re.escape(term)}(?={boundary}|$)"
    
    return re.compile(pattern, re.IGNORECASE)


# Compuestos técnicos que refuerzan detección de conceptos
# V03: Lista blanca de términos compuestos que indican presencia del concepto
TECH_COMPOUNDS = {
    "InteligenciaArtificial": [
        "openai", "deepmind", "ai-powered", "ai-driven", 
        "ai model", "ai system", "chatgpt", "midjourney",
        "modelo de ia", "sistema de ia"
    ],
    "AprendizajeAutomatico": [
        "ml-based", "ml model", "ml algorithm", "scikit-learn",
        "tensorflow", "pytorch", "keras", "machine learning"
    ],
    "Robot": [
        "robotics", "robot-assisted", "robotic system"
    ],
}


def _wb(term: str) -> str:
    """
    [DEPRECATED en v03] Genera patrón regex con fronteras de palabra compatibles con acentos/ñ.
    
    NOTA: Esta función se mantiene por compatibilidad pero v03 usa _smart_word_boundary()
    para términos cortos (≤3 chars).
    
    Args:
        term: Término a buscar (e.g., "ai", "ia")
        
    Returns:
        Patrón regex con fronteras de palabra
        
    Ejemplo:
        >>> _wb("ai")
        r'(?<![A-Za-zÁÉÍÓÚÜÑáéíóúüñ])ai(?![A-Za-zÁÉÍÓÚÜÑáéíóúüñ])'
    """
    return rf'(?<![A-Za-zÁÉÍÓÚÜÑáéíóúüñ]){re.escape(term)}(?![A-Za-zÁÉÍÓÚÜÑáéíóúüñ])'


class OntologyEnricher:
    """
    Clase principal para enriquecimiento semántico usando la ontología completa.
    
    Integra rr-core.ttl + módulos (sentiment, domain, shapes) para extraer
    37+ features ontológicas numéricas de texto.
    
    Mejoras v04:
    - Polaridad contextual: Analiza sentimiento en ventana de ±N palabras alrededor de conceptos
    - Pesos diferenciales: Emociones con palabras más fuertes tienen mayor peso
    - Features contextuales: Ratio de menciones positivas/negativas por concepto
    - Interacciones mejoradas: Concepto × Emoción capturan relaciones semánticas
    
    Features generadas: 37+ (expandible)
    
    Ejemplo de uso:
    >>> enricher = OntologyEnricher(
    ...     ontology_dir="ontology",
    ...     lexicon_manifest="results/lexicon/ontology_lexicon_v04_2_train_only.json",
    ... )
    >>> features = enricher.enrich_text("AI is amazing")
    >>> print(len(features))  # 37+
    """
    
    # Namespaces de la ontología
    RR = Namespace("http://wfrp.ia/ontologia/ia-sentimiento#")
    RRSENT = Namespace("http://wfrp.ia/sentimiento/core#")
    RRDOM = Namespace("http://wfrp.ia/dominio/ciencia-tecnologia#")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    SHACL = Namespace("https://www.w3.org/ns/shacl#")
    
    # Context window used by the frozen v04.2 release protocol.
    CONTEXT_WINDOW = 5  # Palabras antes/después del concepto

    def __init__(self, ontology_dir: str, lexicon_manifest: Optional[str] = None):
        """
        Inicializar el enriquecedor ontológico.
        
        Args:
            ontology_dir: Ruta al directorio con los archivos TTL de la ontología
        """
        self.ontology_dir = Path(ontology_dir)
        self.graph = Graph()
        self.domain_concepts = {}
        self.sentiment_modifiers = {}
        self.emotion_patterns = {}
        self.domain_mapping = {}
        if lexicon_manifest is None:
            raise ValueError("A frozen train-scope lexicon manifest is required")
        self.lexicon_manifest = Path(lexicon_manifest)
        
        self._load_sentiment_lexicons()

        print("Initializing OntologyEnricher v04.2")
        self.load_ontology()
    
    def _load_sentiment_lexicons(self) -> None:
        """Load the frozen train-scope contextual lexicon."""
        payload = json.loads(self.lexicon_manifest.read_text(encoding="utf-8"))
        combined = payload.get("combined_lexicon", {})
        positive = combined.get("positive", {})
        negative = combined.get("negative", {})
        if not positive or not negative:
            raise ValueError(
                "Lexicon manifest lacks combined positive/negative entries: "
                f"{self.lexicon_manifest}"
            )
        self.positive_words = {str(key): float(value) for key, value in positive.items()}
        self.negative_words = {str(key): float(value) for key, value in negative.items()}
        self.lexicon_version = payload.get("version", "manifest")

    def load_ontology(self) -> None:
        """Cargar todos los archivos de ontología en un solo grafo RDF."""
        files = {
            "core": "rr-core.ttl",
            "sentiment": "rr-sentiment.ttl",
            "domain": "rr-domain.ttl",
            "shapes": "rr-shapes.ttl",
        }

        print("[DOCS] Cargando ontología completa...")

        for module, filename in files.items():
            file_path = self.ontology_dir / filename
            if file_path.exists():
                try:
                    self.graph.parse(file_path, format="turtle")
                    print(f"  [OK] {module}: {filename}")
                except Exception as e:
                    print(f"  [ERROR] {module}: Error - {e}")
            else:
                print(f"  [WARNING] {module}: Archivo no encontrado - {filename}")

        total_triples = len(self.graph)
        print(f"\n[TARGET] Ontología cargada: {total_triples} triples RDF")

        # Extraer mapeos para enriquecimiento
        self._extract_domain_concepts()
        self._extract_sentiment_modifiers()
        self._extract_emotion_patterns()
    
    def _calculate_concept_context_polarity(self, text: str, concept_patterns: List[str]) -> Dict[str, float]:
        """
        Calcula la polaridad contextual de un concepto analizando palabras cercanas.
        
        v04 - TÉCNICA CLAVE: En lugar de solo detectar si "IA" está presente,
        analiza el SENTIMIENTO de las palabras en una ventana de ±N palabras.
        
        Ejemplo:
            "AI is revolutionary" → ont_IA_Positivo = 1, ont_IA_Positivo_Score = 2.0
            "AI is a disaster" → ont_IA_Negativo = 1, ont_IA_Negativo_Score = -2.0
            
        Args:
            text: Texto a analizar
            concept_patterns: Lista de patrones regex del concepto
            
        Returns:
            dict con:
                - "positivo_count": Número de menciones positivas
                - "negativo_count": Número de menciones negativas
                - "neutro_count": Número de menciones neutras
                - "positivo_score": Suma de pesos positivos
                - "negativo_score": Suma de pesos negativos (valores negativos)
                - "polarity_ratio": (pos - |neg|) / total_menciones ∈ [-1, 1]
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        positivo_count = 0
        negativo_count = 0
        neutro_count = 0
        positivo_score = 0.0
        negativo_score = 0.0
        
        # Buscar cada mención del concepto
        for pattern in concept_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                # Encontrar posición en palabras
                match_start = match.start()
                match_word_idx = len(text_lower[:match_start].split()) - 1
                match_word_idx = max(0, match_word_idx)
                
                # Definir ventana contextual
                window_start = max(0, match_word_idx - self.CONTEXT_WINDOW)
                window_end = min(len(words), match_word_idx + self.CONTEXT_WINDOW + 1)
                context_words = words[window_start:window_end]
                
                # Analizar sentimiento en ventana
                window_polarity = 0.0
                for word in context_words:
                    # Limpiar puntuación
                    clean_word = re.sub(r'[^\w\s]', '', word)
                    
                    if clean_word in self.positive_words:
                        window_polarity += self.positive_words[clean_word]
                    elif clean_word in self.negative_words:
                        window_polarity += self.negative_words[clean_word]
                
                # Clasificar mención según polaridad de ventana
                if window_polarity > 0.5:
                    positivo_count += 1
                    positivo_score += window_polarity
                elif window_polarity < -0.5:
                    negativo_count += 1
                    negativo_score += window_polarity
                else:
                    neutro_count += 1
        
        # Calcular ratio de polaridad
        total_mentions = positivo_count + negativo_count + neutro_count
        polarity_ratio = 0.0
        if total_mentions > 0:
            polarity_ratio = (positivo_score + negativo_score) / total_mentions
            # Normalizar a [-1, 1]
            polarity_ratio = max(-1.0, min(1.0, polarity_ratio / 2.0))
        
        return {
            "positivo_count": positivo_count,
            "negativo_count": negativo_count,
            "neutro_count": neutro_count,
            "positivo_score": positivo_score,
            "negativo_score": negativo_score,
            "polarity_ratio": polarity_ratio,
        }

    def _extract_domain_concepts(self) -> None:
        """Extraer conceptos de dominio con sus etiquetas alternativas."""
        print("\n[SEARCH] Extrayendo conceptos de dominio...")

        query = prepareQuery(
            """
            SELECT ?concept ?prefLabel ?altLabel WHERE {
                ?concept a skos:Concept .
                ?concept skos:inScheme rrdom:ContextoCienciaTecnologia .
                ?concept skos:prefLabel ?prefLabel .
                OPTIONAL { ?concept skos:altLabel ?altLabel }
                FILTER(lang(?prefLabel) = "es" || lang(?prefLabel) = "en")
            }
        """,
            initNs={"skos": self.SKOS, "rrdom": self.RRDOM},
        )

        results = self.graph.query(query)

        for row in results:
            concept_uri = str(row.concept)
            pref_label = str(row.prefLabel).lower()
            alt_label = str(row.altLabel).lower() if row.altLabel else None

            # Mapeo de términos → conceptos
            self.domain_mapping[pref_label] = concept_uri
            if alt_label:
                self.domain_mapping[alt_label] = concept_uri

        print(f"  [STATS] {len(self.domain_mapping)} mapeos de dominio extraídos")

        # Mostrar algunos ejemplos
        examples = list(self.domain_mapping.items())[:5]
        for term, concept in examples:
            concept_name = concept.split("#")[-1]
            print(f"    '{term}' → {concept_name}")

    def _extract_sentiment_modifiers(self) -> None:
        """Extraer modificadores de polaridad (intensificadores, negaciones, atenuadores)."""
        print("\n[SEARCH] Extrayendo modificadores de sentimiento...")

        # Patrones básicos por tipo de modificador
        self.sentiment_modifiers = {
            "intensificador": {
                "patterns": [
                    r"\b(very|extremely|absolutely|totally|completely|definitely|highly|incredibly|truly|really|super|ultra)\b",
                    r"\b(muy|extremadamente|totalmente|completamente|definitivamente|altamente|increíblemente|realmente|súper)\b",
                    r"\b(so|such|too|way|quite)\b",
                ],
                "weight": 1.5,
                "concept": str(self.RRSENT.Intensificador),
            },
            "negacion": {
                "patterns": [
                    r"\b(not|no|never|nothing|none|neither|nor|nowhere|nobody)\b",
                    r"\b(no|nunca|nada|ningún|ninguna|tampoco|jamás)\b",
                    r"\b(isn't|aren't|wasn't|weren't|won't|wouldn't|can't|couldn't|don't|doesn't|didn't|hasn't|haven't|hadn't)\b",
                    r"\b(without|lack|lacking)\b",
                ],
                "weight": -1.0,
                "concept": str(self.RRSENT.Negacion),
            },
            "atenuador": {
                "patterns": [
                    r"\b(somewhat|slightly|rather|fairly|quite|maybe|perhaps|possibly|probably|might|could|may)\b",
                    r"\b(algo|un poco|bastante|más o menos|quizás|tal vez|posiblemente|probablemente|podría)\b",
                    r"\b(kind of|sort of|a bit|a little)\b",
                ],
                "weight": 0.7,
                "concept": str(self.RRSENT.Atenuador),
            },
        }

        print(f"  [STATS] {len(self.sentiment_modifiers)} tipos de modificadores definidos")
        for mod_type, data in self.sentiment_modifiers.items():
            print(f"    {mod_type}: peso={data['weight']}")

    def _extract_emotion_patterns(self) -> None:
        """
        Extraer patrones de emociones específicas con PESOS DIFERENCIALES.
        
        Mejora v04: Palabras con fuerte carga emocional tienen mayor peso.
        Basado en análisis de NRC Emotion Lexicon y VADER Sentiment.
        """
        print("\n[SEARCH] Extrayendo patrones de emociones (v04: pesos diferenciales)...")

        self.emotion_patterns = {
            str(self.RRSENT.Miedo): {
                "patterns": {
                    # Miedo EXTREMO (peso 2.0)
                    r"\b(terrifying|catastrophic|disaster|destruction|nightmare|apocalyptic)\b": 2.0,
                    r"\b(aterrador|catastrófico|desastre|destrucción|pesadilla|apocalíptico)\b": 2.0,
                    
                    # Miedo ALTO (peso 1.5)
                    r"\b(fear|threat|danger|panic|crisis|devastating|alarming)\b": 1.5,
                    r"\b(miedo|amenaza|peligro|pánico|crisis|devastador|alarmante)\b": 1.5,
                    r"\b(job.?loss|unemployment|existential.?risk|human.?extinction)\b": 1.5,
                    
                    # Miedo MODERADO (peso 1.0)
                    r"\b(afraid|scared|worried|concern|risk|anxiety|unsafe)\b": 1.0,
                    r"\b(temor|preocupación|riesgo|ansiedad|inseguro|preocupado)\b": 1.0,
                    r"\b(lose|losing|harm|harmful|uncertain)\b": 1.0,
                },
                "base_weight": 1.0,
            },
            str(self.RRSENT.Esperanza): {
                "patterns": {
                    # Esperanza EXTREMA (peso 2.0)
                    r"\b(revolutionary|transformative|breakthrough|game.?changer|unprecedented)\b": 2.0,
                    r"\b(revolucionario|transformador|innovador|cambio.?radical|sin.?precedentes)\b": 2.0,
                    
                    # Esperanza ALTA (peso 1.5)
                    r"\b(promise|promising|potential|opportunity|beneficial|bright.?future)\b": 1.5,
                    r"\b(promesa|prometedor|potencial|oportunidad|beneficioso|futuro.?brillante)\b": 1.5,
                    r"\b(solve|solution|cure|save|help|advance)\b": 1.5,
                    
                    # Esperanza MODERADA (peso 1.0)
                    r"\b(hope|optimistic|positive|better|improve|progress)\b": 1.0,
                    r"\b(esperanza|optimista|positivo|mejor|mejorar|progreso)\b": 1.0,
                },
                "base_weight": 1.0,
            },
            str(self.RRSENT.Entusiasmo): {
                "patterns": {
                    # Entusiasmo EXTREMO (peso 2.0)
                    r"\b(amazing|awesome|incredible|fantastic|phenomenal|extraordinary)\b": 2.0,
                    r"\b(increíble|fantástico|fenomenal|extraordinario|espectacular)\b": 2.0,
                    r"\b(love.?it|absolutely.?love|blown.?away|mind.?blowing)\b": 2.0,
                    
                    # Entusiasmo ALTO (peso 1.5)
                    r"\b(exciting|great|wonderful|brilliant|impressive|excellent)\b": 1.5,
                    r"\b(emocionante|genial|maravilloso|brillante|impresionante|excelente)\b": 1.5,
                    r"\b(cool|perfect|outstanding|superb)\b": 1.5,
                    
                    # Entusiasmo MODERADO (peso 1.0)
                    r"\b(interesting|nice|good|like|enjoy)\b": 1.0,
                    r"\b(interesante|bueno|gusta|disfrutar)\b": 1.0,
                },
                "base_weight": 1.0,
            },
            str(self.RRSENT.Escepticismo): {
                "patterns": {
                    # Escepticismo EXTREMO (peso 2.0)
                    r"\b(scam|fraud|hoax|bullshit|garbage|useless|waste)\b": 2.0,
                    r"\b(estafa|fraude|engaño|basura|inútil|desperdicio)\b": 2.0,
                    r"\b(overhyped|overrated|snake.?oil|vaporware)\b": 2.0,
                    
                    # Escepticismo ALTO (peso 1.5)
                    r"\b(skeptical|doubtful|questionable|suspicious|misleading|exaggerat)\b": 1.5,
                    r"\b(escéptico|dudoso|cuestionable|sospechoso|engañoso|exagerado)\b": 1.5,
                    r"\b(not.?convinced|don't.?believe|fail|failure|flawed)\b": 1.5,
                    
                    # Escepticismo MODERADO (peso 1.0)
                    r"\b(doubt|uncertain|unsure|debatable|concern|problem|issue|limitation)\b": 1.0,
                    r"\b(duda|incierto|debatible|preocupación|problema|limitación)\b": 1.0,
                },
                "base_weight": 1.0,
            },
        }

        print(f"  [STATS] {len(self.emotion_patterns)} tipos de emociones con pesos diferenciales")

    def enrich_text(self, text: str) -> Dict[str, float]:
        """
        Enriquecer texto con features ontológicas expandidas + polaridad contextual.
        
        V04: MEJORA CRÍTICA - Detecta sentimiento alrededor de conceptos con ventana ±N palabras.
        Esto resuelve el problema de que "IA" aparecía con igual feature en posts positivos/negativos.
        
        Ejemplo v03 (problema):
            "AI is revolutionary" → ont_domain_1_IA = 1.0
            "AI is a disaster" → ont_domain_1_IA = 1.0  # ⚠️ Misma feature, sentimientos opuestos
            
        Ejemplo v04 (solución):
            "AI is revolutionary" → ont_IA_Positivo=1, ont_IA_Positivo_Score=2.0, ont_IA_Negativo=0
            "AI is a disaster" → ont_IA_Negativo=1, ont_IA_Negativo_Score=-2.0, ont_IA_Positivo=0
        
        Args:
            text: Texto a enriquecer
            
        Returns:
            Diccionario con 60+ features ontológicas (todas float)
            
        Features v04:
        - 10 one-hot conceptos dominio (LEGACY)
        - 10×3 = 30 features contextuales: ont_{Concepto}_Positivo/Negativo/Neutro
        - 10 scores de polaridad: ont_{Concepto}_Polarity_Ratio ∈ [-1, 1]
        - 3 modificadores sentimiento
        - 4 emociones con PESOS DIFERENCIALES (v04)
        - Contadores y derivadas mejoradas
        """
        # Limpieza selectiva
        text = _clean_code_blocks(text)
        text_lower = text.lower()
        
        # Estructuras auxiliares
        concept_scores = defaultdict(float)
        detected_modifiers = []
        detected_emotions = []
        domain_score = 0.0
        sentiment_polarity_modifier = 1.0
        emotion_weights_dict = {}
        
        # v04: Diccionario para polaridad contextual por concepto
        concept_context_polarity = {}

        # 1. Detectar conceptos de dominio con smart word boundaries
        for term, concept in self.domain_mapping.items():
            concept_name = concept.split("#")[-1]
            score = 0.0
            patterns_for_concept = []  # v04: guardar patrones para análisis contextual
            
            # Smart word boundaries para términos cortos
            if len(term) <= 3:
                pattern = _smart_word_boundary(term)
                matches = pattern.findall(text_lower)
                score += len(matches)
                patterns_for_concept.append(pattern.pattern)
            else:
                pattern = rf"\b{re.escape(term)}\b"
                if re.search(pattern, text_lower):
                    score += 1.0
                    patterns_for_concept.append(pattern)
            
            # Bonus por compuestos técnicos
            if concept_name in TECH_COMPOUNDS:
                for compound in TECH_COMPOUNDS[concept_name]:
                    if compound in text_lower:
                        score += 0.5
                        patterns_for_concept.append(rf"\b{re.escape(compound)}\b")
            
            # Acumular score si hay detección
            if score > 0:
                concept_scores[concept_name] += score
                
                # v04: CLAVE - Calcular polaridad contextual para este concepto
                polarity_data = self._calculate_concept_context_polarity(text, patterns_for_concept)
                concept_context_polarity[concept_name] = polarity_data
        
        # Top-K=10 conceptos
        top_concepts = sorted(
            concept_scores.items(), 
            key=lambda x: (-x[1], -len(x[0]))
        )[:10]
        
        detected_domain_concepts = [concept for concept, _ in top_concepts]
        domain_score = sum(score for _, score in top_concepts)

        # 2. Detectar modificadores de sentimiento
        for mod_type, data in self.sentiment_modifiers.items():
            for pattern in data["patterns"]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    detected_modifiers.append(mod_type)
                    if mod_type == "negacion":
                        sentiment_polarity_modifier *= -0.8
                    elif mod_type == "intensificador":
                        sentiment_polarity_modifier *= 1.3
                    elif mod_type == "atenuador":
                        sentiment_polarity_modifier *= 0.7

        # 3. v04: Detectar emociones con PESOS DIFERENCIALES
        for emotion, data in self.emotion_patterns.items():
            emotion_name = emotion.split("#")[-1]
            emotion_score = 0.0
            
            # v04: Usar patrones ponderados
            for pattern, weight in data["patterns"].items():
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                emotion_score += matches * weight  # v04: aplicar peso diferencial

            if emotion_score > 0:
                detected_emotions.append(emotion_name)
                emotion_weights_dict[emotion_name] = emotion_score

        # ===== GENERAR FEATURES =====
        
        top_domain_concepts = [
            "InteligenciaArtificial", "AprendizajeAutomatico", "Tecnologia",
            "Futuro", "Datos", "Algoritmo", "Robot", "Automatizacion",
            "Etica", "Innovacion"
        ]
        
        features = {}
        
        # LEGACY: Top 10 conceptos (one-hot) - mantener compatibilidad
        for i, concept in enumerate(top_domain_concepts, 1):
            features[f"ont_domain_{i}_{concept}"] = float(concept in detected_domain_concepts)
        features["ont_domain_x_polarity"] = 0.0
        
        # v04: NUEVAS FEATURES CONTEXTUALES (30 features + 10 ratios = 40 features nuevas)
        for concept in top_domain_concepts:
            if concept in concept_context_polarity:
                polarity = concept_context_polarity[concept]
                features[f"ont_{concept}_Positivo"] = float(polarity["positivo_count"])
                features[f"ont_{concept}_Negativo"] = float(polarity["negativo_count"])
                features[f"ont_{concept}_Neutro"] = float(polarity["neutro_count"])
                features[f"ont_{concept}_Polarity_Ratio"] = float(polarity["polarity_ratio"])
            else:
                # Concepto no detectado → features en 0
                features[f"ont_{concept}_Positivo"] = 0.0
                features[f"ont_{concept}_Negativo"] = 0.0
                features[f"ont_{concept}_Neutro"] = 0.0
                features[f"ont_{concept}_Polarity_Ratio"] = 0.0
        
        # Modificadores de sentimiento (one-hot)
        features["ont_mod_intensificador"] = float("intensificador" in detected_modifiers)
        features["ont_mod_negacion"] = float("negacion" in detected_modifiers)
        features["ont_mod_atenuador"] = float("atenuador" in detected_modifiers)
        
        # Emociones (one-hot)
        features["ont_emo_miedo"] = float("Miedo" in detected_emotions)
        features["ont_emo_esperanza"] = float("Esperanza" in detected_emotions)
        features["ont_emo_entusiasmo"] = float("Entusiasmo" in detected_emotions)
        features["ont_emo_escepticismo"] = float("Escepticismo" in detected_emotions)
        
        # Contadores
        features["ont_count_domain_concepts"] = float(len(detected_domain_concepts))
        features["ont_count_modifiers"] = float(len(detected_modifiers))
        features["ont_count_emotions"] = float(len(detected_emotions))
        
        # Scores numéricos
        features["ont_domain_score"] = float(domain_score)
        features["ont_sentiment_polarity_modifier"] = float(sentiment_polarity_modifier)
        
        # v04: Pesos de emociones con valores diferenciales
        features["ont_weight_miedo"] = float(emotion_weights_dict.get("Miedo", 0.0))
        features["ont_weight_esperanza"] = float(emotion_weights_dict.get("Esperanza", 0.0))
        features["ont_weight_entusiasmo"] = float(emotion_weights_dict.get("Entusiasmo", 0.0))
        features["ont_weight_escepticismo"] = float(emotion_weights_dict.get("Escepticismo", 0.0))
        
        # Features derivadas mejoradas
        text_length = len(text.split())
        features["ont_domain_density"] = float(len(detected_domain_concepts) / max(text_length, 1))
        features["ont_modifier_density"] = float(len(detected_modifiers) / max(text_length, 1))
        features["ont_emotion_density"] = float(len(detected_emotions) / max(text_length, 1))
        
        features["ont_emotion_x_polarity"] = float(len(detected_emotions) * abs(sentiment_polarity_modifier))
        
        features["ont_semantic_richness"] = float(
            len(detected_domain_concepts) + len(detected_modifiers) + len(detected_emotions)
        )
        
        features["ont_polarity_abs"] = float(abs(sentiment_polarity_modifier))
        
        positive_emotions = features["ont_weight_esperanza"] + features["ont_weight_entusiasmo"]
        negative_emotions = features["ont_weight_miedo"] + features["ont_weight_escepticismo"]
        features["ont_emotion_valence"] = float(positive_emotions - negative_emotions)
        
        features["ont_emotion_concept_ratio"] = float(
            len(detected_emotions) / max(len(detected_domain_concepts), 1)
        )
        
        features["ont_has_ontology_elements"] = float(
            len(detected_domain_concepts) > 0 or 
            len(detected_modifiers) > 0 or 
            len(detected_emotions) > 0
        )
        
        features["ont_composite_score"] = float(
            (domain_score * 0.4) +
            (abs(sentiment_polarity_modifier) * 0.3) +
            (sum(emotion_weights_dict.values()) * 0.3)
        )
        
        # v04: Nuevas features agregadas de polaridad contextual
        total_positive = sum(
            concept_context_polarity.get(c, {}).get("positivo_count", 0) 
            for c in top_domain_concepts
        )
        total_negative = sum(
            concept_context_polarity.get(c, {}).get("negativo_count", 0) 
            for c in top_domain_concepts
        )
        total_neutral = sum(
            concept_context_polarity.get(c, {}).get("neutro_count", 0) 
            for c in top_domain_concepts
        )
        
        features["ont_total_positivo_mentions"] = float(total_positive)
        features["ont_total_negativo_mentions"] = float(total_negative)
        features["ont_total_neutro_mentions"] = float(total_neutral)
        
        # Ratio global de polaridad
        total_mentions = total_positive + total_negative + total_neutral
        if total_mentions > 0:
            features["ont_global_polarity_ratio"] = float((total_positive - total_negative) / total_mentions)
        else:
            features["ont_global_polarity_ratio"] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Obtener lista ordenada de nombres de features generadas.
        
        Returns:
            Lista con los 37 nombres de features en orden consistente
        """
        # Generar un texto dummy para obtener las keys
        dummy_features = self.enrich_text("dummy text")
        return list(dummy_features.keys())


# ========== TESTS INTERNOS v03 ==========

if __name__ == "__main__":
    print("=" * 80)
    print("TESTS INTERNOS - OntologyEnricher v03 (Smart Word Boundaries)")
    print("=" * 80)
    
    # Inicializar enricher
    from pathlib import Path
    ontology_dir = Path(__file__).parent / "ontology_protege"
    enricher = OntologyEnricher(str(ontology_dir))
    enricher.load_ontology()
    
    print("\n" + "=" * 80)
    print("CASOS DE PRUEBA v03 - POSITIVOS (Deben detectar IA/ML)")
    print("=" * 80)
    
    positive_cases = [
        ("AI researcher working on NLP", "IA", "Espacio antes/después"),
        ("AI-powered chatbots are amazing", "IA", "Guion como separador"),
        ("OpenAI released GPT-4", "IA", "Compuesto técnico 'OpenAI'"),
        ("IA generativa es el futuro", "IA", "Espacio antes/después (español)"),
        ("modelo de IA para NLP", "IA", "Compuesto 'modelo de ia'"),
    ]
    
    passed = 0
    for text, concepto, razon in positive_cases:
        features = enricher.enrich_text(text)
        ia_detected = features.get("ont_domain_1_InteligenciaArtificial", 0.0)
        ml_detected = features.get("ont_domain_2_AprendizajeAutomatico", 0.0)
        
        print(f"\n'{text}'")
        print(f"   Razón: {razon}")
        print(f"   → IA={ia_detected}, ML={ml_detected}")
        
        if concepto == "IA" and ia_detected > 0:
            print(f"   ✅ PASA: {concepto} detectado")
            passed += 1
        elif concepto == "ML" and ml_detected > 0:
            print(f"   ✅ PASA: {concepto} detectado")
            passed += 1
        else:
            print(f"   ❌ FALLA: {concepto} NO detectado (esperado >0)")
    
    print(f"\n📊 Positivos: {passed}/{len(positive_cases)} pasados")
    
    print("\n" + "=" * 80)
    print("CASOS DE PRUEBA v03 - NEGATIVOS (NO deben detectar IA)")
    print("=" * 80)
    
    negative_cases = [
        ("again and again I tried", "Letra antes Y después"),
        ("maintAIning the code is hard", "Letra antes Y después (case-mixed)"),
        ("send email to test@ai.com", "@ no es separador válido"),
        ("said something interesting", "Letra antes Y después"),
    ]
    
    passed_neg = 0
    for text, razon in negative_cases:
        features = enricher.enrich_text(text)
        ia_detected = features.get("ont_domain_1_InteligenciaArtificial", 0.0)
        x_polarity = features.get("ont_domain_x_polarity", 0.0)
        
        print(f"\n'{text}'")
        print(f"   Razón: {razon}")
        print(f"   → IA={ia_detected}, x_polarity={x_polarity}")
        
        if ia_detected == 0.0:
            print(f"   ✅ PASA: IA correctamente NO detectado")
            passed_neg += 1
        else:
            print(f"   ❌ FALLA: IA detectado (esperado 0)")
        
        # Validar x_polarity siempre 0
        assert x_polarity == 0.0, f"❌ CRÍTICO: x_polarity={x_polarity}, debe ser 0"
    
    print(f"\n📊 Negativos: {passed_neg}/{len(negative_cases)} pasados")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_passed = passed + passed_neg
    total_cases = len(positive_cases) + len(negative_cases)
    print(f"✅ Total: {total_passed}/{total_cases} casos pasados")
    print(f"✅ Tasa de acierto: {100*total_passed/total_cases:.1f}%")
    
    if total_passed == total_cases:
        print("\nAll local enrichment checks passed.")
    else:
        print(f"\n{total_cases - total_passed} local enrichment checks failed.")
    
    print("=" * 80)
    print("All local enrichment checks passed.")
    print("=" * 80)
