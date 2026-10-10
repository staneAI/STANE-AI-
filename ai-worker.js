
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

let generator = null;
let loading = false;
let generating = false;

env.allowLocalModels = false;
env.useBrowserCache = true;

const RULES = `Tu es STANE AI, assistant bilingue français-allemand.
Réponds directement à la demande, dans la langue de l'utilisateur.
Utilise un français naturel et grammaticalement correct.
En allemand, respecte les articles, les cas, la conjugaison et l'ordre des mots.
Ne fabrique pas de contexte, de recherche ou de source.
Sois concis par défaut, mais développe si la question le demande.
Pour les questions scolaires, explique clairement et distingue les faits des suppositions.`;

function quickReply(question) {
  const q = question.trim().toLocaleLowerCase()
    .normalize("NFC")
    .replace(/[!.?,;:]+$/g, "");

  const french = [
    "bonjour", "bonjour stane", "salut", "salut stane",
    "bonsoir", "bonsoir stane", "coucou"
  ];

  const german = [
    "hallo", "hallo stane", "guten tag",
    "guten morgen", "guten abend"
  ];

  if (french.includes(q)) {
    if (q.startsWith("bonsoir")) {
      return "Bonsoir ! Comment puis-je vous aider ?";
    }
    if (q === "coucou") {
      return "Coucou ! Que puis-je faire pour vous ?";
    }
    return "Bonjour ! Ravi de vous retrouver. Comment puis-je vous aider aujourd’hui ?";
  }

  if (german.includes(q)) {
    return "Hallo! Wie kann ich dir heute helfen?";
  }

  if (q === "hello" || q === "hi") {
    return "Hello! How can I help you today?";
  }

  if (["merci", "merci stane", "danke", "danke stane"].includes(q)) {
    return q.startsWith("danke")
      ? "Gern geschehen! Wie kann ich dir weiterhelfen?"
      : "Avec plaisir ! Que puis-je faire d’autre pour vous ?";
  }

  return null;
}

self.onmessage = async (event) => {
  const d = event.data || {};

  if (d.type === "load") {
    if (generator || loading) return;

    loading = true;
    self.postMessage({
      type: "status",
      message: "Chargement du modèle IA. Internet est nécessaire au premier téléchargement.",
      level: "wait"
    });

    try {
      generator = await pipeline(
        "text-generation",
        "onnx-community/Qwen2.5-0.5B-Instruct",
        {
          dtype: "q4",
          progress_callback: (p) => {
            if (p.status === "progress" && p.file) {
              self.postMessage({
                type: "status",
                message: `Téléchargement : ${p.file} (${Math.round(p.progress || 0)} %).`,
                level: "wait"
              });
            }
          }
        }
      );

      self.postMessage({
        type: "ready",
        message: "Modèle chargé. STANE AI est prêt."
      });
    } catch (error) {
      self.postMessage({
        type: "error",
        message: "Chargement impossible. Vérifiez Internet et les ressources de l’appareil."
      });
    } finally {
      loading = false;
    }
    return;
  }

  if (d.type !== "generate") return;

  const question = String(d.question || "").trim().slice(0, 1200);

  if (!question) {
    self.postMessage({
      type: "answer",
      text: "Écrivez votre question pour commencer."
    });
    return;
  }

  // Fonctions locales : aucune attente du modèle pour les demandes reconnues.
  const quick = quickReply(question);
  if (quick) {
    self.postMessage({ type: "answer", text: quick });
    return;
  }

  if (!generator) {
    self.postMessage({
      type: "error",
      message: "Cette question nécessite le modèle IA. Cliquez sur « Charger le modèle IA », puis réessayez."
    });
    return;
  }

  if (generating) {
    self.postMessage({
      type: "error",
      message: "Une réponse est déjà en cours. Veuillez patienter."
    });
    return;
  }

  generating = true;

  try {
    const mode = d.mode || "general";
    let instructions = RULES;

    if (mode === "pedago") {
      instructions += ` Tu es aussi un assistant pédagogique. Structure les explications, exercices et corrigés. N'invente pas de références officielles.`;
    }

    if (mode === "doc") {
      instructions += ` Analyse uniquement le texte fourni. Distingue les informations explicites des interprétations.`;
    }

    if (mode === "doc" && d.context) {
      instructions += `\nTexte fourni :\n${String(d.context).slice(0, 2500)}`;
    }

    const prompt =
      `System: ${instructions}\n` +
      `User: ${question}\n` +
      `Assistant:`;

    const result = await generator(prompt, {
      max_new_tokens: 112,
      do_sample: false,
      repetition_penalty: 1.08
    });

    let answer = result?.[0]?.generated_text || "";

    if (answer.startsWith(prompt)) {
      answer = answer.slice(prompt.length);
    } else {
      const marker = answer.lastIndexOf("Assistant:");
      if (marker >= 0) answer = answer.slice(marker + 10);
    }

    answer = answer.trim();

    self.postMessage({
      type: "answer",
      text: answer || "Je n’ai pas bien compris. Pouvez-vous reformuler ?"
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      message: "La génération a échoué. Essayez une question plus courte."
    });
  } finally {
    generating = false;
  }
};
