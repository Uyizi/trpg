"""Disposable HTTP transport. Parent enforces absolute time/cancel deadlines."""
import json
import sys
from .api_runtime import ApiRuntime
from .domain import RuntimeProfile
from .runtime import NotSent,ResponseError
from .worker_protocol import exit_with_parent


def main():
    exit_with_parent()
    try:
        args=json.load(sys.stdin);runtime=ApiRuntime(RuntimeProfile.model_validate(args['profile']))
        result={'kind':'result','result':runtime._generate_blocking(args['request'])}
    except NotSent as exc:result={'kind':'not_sent','message':str(exc)}
    except ResponseError as exc:result={'kind':'response_error','message':str(exc),'evidence':exc.evidence}
    except Exception as exc:result={'kind':'unknown','message':type(exc).__name__+': '+str(exc)}
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
