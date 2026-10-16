"""Conservative request routing; mentioning a tool is not a request to use it."""
import re


# This removes request scaffolding, not subject vocabulary. It is deliberately
# small: lexical overlap is a cheap rejection gate, never a fact checker.
_SCAFFOLD_ZH = re.compile(
    r'请你|帮我|麻烦|请|根据|结合|解释|比较|对比|综述|查找|查询|搜索|检索|查一下|搜一下|'
    r'本地文档|本地资料|知识库|上传的文档|这份文档|论文|文献|资料|'
    r'最新|今天|昨天|今年|最近|目前|现在|有哪些|是什么|什么是|哪个|如何|怎么|一下')
_SCAFFOLD_EN = frozenset(
    'a an the is are was were be to of for in on at and or with without '
    'please find search browse lookup look up explain summarize summarise '
    'papers paper literature documents document about based compare review latest current '
    'today recently this year what which who where when how'.split())
_LEXEMES = re.compile(r'[a-z0-9]+(?:[_.-][a-z0-9]+)*|[\u3400-\u9fff]+', re.I)
_YEAR = re.compile(r'(?<!\d)((?:19|20)\d{2})(?!\d)')
_ANNUAL_EVENTS = re.compile(
    r'大事(?:记)?|重大事件|重要事件|年度(?:新闻|事件|回顾)|'
    r'(?:国际|世界|全球|国内外|国内)新闻|新闻(?:回顾|盘点)|时事|'
    r'\b(?:major events|year in review|top news|annual news|news roundup)\b', re.I)


def _named_variant(name, candidates):
    # A family name can denote a suffixed variant; numeric prefixes cannot.
    return any(candidate == name or candidate.startswith((name + '-', name + '_'))
               for candidate in candidates)


def content_query(query):
    """Remove common request/freshness words before lexical candidate search."""
    text = _SCAFFOLD_ZH.sub(' ', query.lower())
    return ' '.join(word for word in _LEXEMES.findall(text)
                    if word not in _SCAFFOLD_EN and word not in {'的', '和', '与', '及'}
                    and not re.fullmatch(r'\d{4}', word))


def relevance_terms(query):
    terms = set()
    for word in _LEXEMES.findall(content_query(query)):
        if '\u3400' <= word[0] <= '\u9fff':
            terms.update(word[i:i + 2] for i in range(len(word) - 1))
            if len(word) == 1:
                terms.add(word)
        else:
            terms.add(word)
    return terms


def identifiers_match(query, text):
    """Versioned names must match a full lexeme, not a nearby version prefix."""
    names = {word for word in _LEXEMES.findall(query.lower())
             if re.search(r'[a-z]', word) and re.search(r'\d', word)}
    evidence = set(_LEXEMES.findall(text.lower()))
    # A comparison can legitimately retrieve a separate document per model.
    return not names or any(_named_variant(name, evidence) for name in names)


def _annual_event_relevance(query, text):
    """Narrow alias for explicit-year event reviews, not any same-year page."""
    years = set(_YEAR.findall(query))
    if not years or not _ANNUAL_EVENTS.search(query):
        return None
    if not years.intersection(_YEAR.findall(text)):
        return False
    event_topic = re.search(r'新闻|大事|重大事件|重要事件|\b(?:news|events)\b', text, re.I)
    review = re.search(r'年度|年终|全年|十大|大事记|重大事件|重要事件|回顾|盘点|'
                       r'\b(?:year in review|annual|roundup|top\s+(?:\d+\s+)?(?:world\s+)?news)\b', text, re.I)
    if not (event_topic and review):
        return False
    # Preserve a user-specified subject: "量子纠错大事" cannot become generic news.
    subject = re.sub(r'(?<!\d)(?:19|20)\d{2}\s*年?', ' ', query)
    subject = _ANNUAL_EVENTS.sub(' ', subject)
    subject = re.sub(r'发生(?:了|过)?|(?:有|有过)?(?:什么|哪些)|值得关注的|主要|'
                     r'\b(?:happened|occurred)\b', ' ', subject, flags=re.I)
    subject = content_query(subject)
    return not subject or relevant_text(subject, text)


def relevant_text(query, text):
    """Reject obvious lexical distractors; passing does not establish support.

    Require half the informative query terms (rounded up). Dense multilingual
    retrieval uses only identifiers_match, since translations need not overlap.
    """
    if not identifiers_match(query, text):
        return False
    annual = _annual_event_relevance(query, text)
    if annual is not None:
        return annual
    terms = relevance_terms(query)
    if not terms:
        return False
    evidence_terms = set(_LEXEMES.findall(text.lower()))
    matched = sum(term in text.lower() if re.search(r'[\u3400-\u9fff]', term)
                  else _named_variant(term, evidence_terms) if re.search(r'[a-z]', term) and re.search(r'\d', term)
                  else term in evidence_terms for term in terms)
    return matched >= (len(terms) + 1) // 2


def web_evidence_guidance(result):
    """Per-turn evidence policy; never turn search failure into a global refusal."""
    status = result['status']
    guidance = '\n本轮联网检索状态：' + status + '。' + result.get('notice', '')
    if status in ('failed', 'empty'):
        guidance += ('本轮未获得可用联网证据。对最新动态、具体年度事件和待核实的实体细节，'
                     '简短说明未核实，不以惯例模板、旧资料或猜测代替结果；'
                     '仍可直接回答有把握的稳定知识、计算和用户给定材料，区分未核实部分。')
    elif status == 'limited':
        guidance += ('目前只有搜索摘要，未取得相关正文；摘要仅作线索。'
                     '对摘要未支持的数字、日期和最新结论说明未核实。')
    guidance += ('候选资料的排名和相似度不是事实置信度；只在内容支持对应结论时引用。'
                 '检索时间不是文章发布时间，不能证明内容属于所问年份或是最新消息。')
    return guidance


def company_fact_question(question):
    """Company identity/background needs sources even without a recency word."""
    company = re.search(r'公司|工作室|厂商|开发商|发行商|\b(?:company|studio|developer|publisher)\b', question, re.I)
    attribute = re.search(r'哪[个家国]|什么(?:公司|地方|时候)|成立|创立|创办|总部|位于|地址|创始人|联合创始人|由谁|'
                          r'\b(?:who|where|when|founded|founder|headquarters)\b', question, re.I)
    challenge = re.search(r'为什么你(?:认为|说)|你(?:刚才|之前)?(?:说错|搞错)|你确定|有何依据|什么依据|来源|出处', question)
    return bool(company and (attribute or challenge))


def retrieval_meta_question(question):
    feature = re.search(r'检索记录|搜索记录|检索清单|来源列表|联网模式|检索功能|搜索功能|/sources|/web', question, re.I)
    discussion = re.search(r'为什么|为何|怎么|如何|什么|何意味|不要|不想|取消|关闭|每[个次轮]|why|how|what', question, re.I)
    return bool(feature and discussion)


def wants_papers(question):
    if retrieval_meta_question(question):
        return False
    if re.search(r'(?:不要|不用|无需|不必|别)(?:搜索|检索|查找|查|找|引用)?(?:论文|文献|资料)', question):
        return False
    topic = re.search(r'论文|文献|研究资料|学术资料|arxiv|\bpapers?\b|\bliterature\b', question, re.I)
    request = re.search(r'查找|查询|搜索|检索|查阅|查一下|找|推荐|列出|有哪些|综述|对比|比较|总结|根据|结合|引用|依据|参考|证据|find|search|review|cite|summari[sz]e|compare', question, re.I)
    return bool(topic and request)


def wants_local_documents(question):
    if retrieval_meta_question(question):
        return False
    source = re.search(r'知识库|本地(?:资料|文档|文件)|上传的(?:资料|文档|文件)|这份(?:资料|文档)|\bdocument\b', question, re.I)
    request = re.search(r'查|搜|找|根据|结合|总结|分析|比较|对比|解释|问|search|find|summari[sz]e|based on', question, re.I)
    return bool(source and request)
