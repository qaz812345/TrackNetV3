"""TrackNet model — re-exported from the original model.py with a golf-aware factory."""

from model import TrackNet  # original unchanged architecture

from tracknet_golf.config import GolfConfig, _tracknet_in_dim


def build_tracknet(cfg: GolfConfig) -> TrackNet:
    """Instantiate TrackNet with correct channel counts for a given golf config."""
    in_dim = _tracknet_in_dim(cfg.bg_mode, cfg.seq_len)
    out_dim = cfg.seq_len
    return TrackNet(in_dim=in_dim, out_dim=out_dim)


__all__ = ["TrackNet", "build_tracknet"]
