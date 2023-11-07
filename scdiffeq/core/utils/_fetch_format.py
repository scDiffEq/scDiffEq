
# from ... import tools

# from autodevice import AutoDevice
# NoneType = type(None)

# def fetch_format(adata, use_key: str, idx=None, N=1, device=AutoDevice()):

#     if isinstance(idx, NoneType):
#         idx = range(len(adata))

#     return tools.fetch(adata[idx], use_key=use_key, device=device)[:, None].expand(
#         -1, N, -1
#     )

