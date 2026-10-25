"""Strict buyer-agent boundary around the team's shared root NLU implementation."""
from nlu import ACTS, MAX_LEN, NLU


class NLULoadError(RuntimeError):
    """Requested model resources could not be loaded; no fallback is used."""


class NLUAdapter:
    """Validate public turns and expose shared inference at full precision."""

    def __init__(self, checkpoint_path, model_name="distilbert-base-uncased", device="cpu",
                 *, local_files_only=False):
        try:
            self._nlu = NLU(checkpoint_path, device, model_name=model_name,
                            local_files_only=local_files_only)
        except Exception as exc:
            raise NLULoadError(
                f"Cannot load NLU checkpoint {checkpoint_path} with {model_name}: {exc}") from exc

    @property
    def resource_metadata(self) -> dict:
        return self._nlu.resource_metadata

    def predict(self, history: list[tuple[str, str]], message: str) -> dict[str, float]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a nonempty seller message")
        if not isinstance(history, list):
            raise ValueError("history must be an oldest-first list of (BUYER|SELLER, text) tuples")
        for turn in history:
            if (not isinstance(turn, tuple) or len(turn) != 2
                    or turn[0] not in ("BUYER", "SELLER") or not isinstance(turn[1], str)):
                raise ValueError("each history turn must be a (BUYER|SELLER, text) tuple")
        return self._nlu._predict_probabilities(history, message)
