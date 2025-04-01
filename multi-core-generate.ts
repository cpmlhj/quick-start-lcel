import dot from 'dotenv'
import { ChatOpenAI } from '@langchain/openai'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { RunnablePassthrough, RunnableMap } from '@langchain/core/runnables'
import { HttpsProxyAgent } from 'https-proxy-agent'

dot.config()
;(async () => {
	const model = new ChatOpenAI({
		model: 'gpt-4',
		configuration: {
			baseURL: process.env.OPENAI_BASE_URL,
			httpAgent: new HttpsProxyAgent(process.env.HTTP_PROXY || '')
		}
	})
	const planner = ChatPromptTemplate.fromTemplate(
		`
         根据以下内容补充生成能适应不同编程语言的提示词: {input}

         ## 注意事项
         1. 主需要润色提示词，无需出现具体的编程语言
         2. 润色后的提示词必须能够适应不同的场景
         3. 选取你认为最合适的一个选项，只需要返回一个
        `
	)
		.pipe(model)
		.pipe(new StringOutputParser())
		.pipe(() =>
			RunnablePassthrough.assign({
				base_question: (output) => output
			})
		)

	// 创建 Java 代码
	const java_chain = ChatPromptTemplate.fromTemplate(
		`
            请根据以下内容生成 Java 代码: {base_question}
            ## 注意事项
            1. 生成的代码必须是可执行的
            2. 生成的代码必须按照 Java 的规范进行编写
            3. 注释必须按照 Java 的规范进行编写
            `
	)
		.pipe(model.bind({ configurable: { model: 'gpt-4o-mini' } }))
		.pipe(new StringOutputParser())

	// 创建 Python 代码
	const python_chain = ChatPromptTemplate.fromTemplate(
		`
            请根据以下内容生成 Python 代码: {base_question}
             ## 注意事项
            1. 生成的代码必须是可执行的
            2. 生成的代码必须按照 python 的规范进行编写
            3. 注释必须按照 python 的规范进行编写
            `
	)
		.pipe(model.bind({ configurable: { model: 'gpt-4o-mini' } }))
		.pipe(new StringOutputParser())

	// 创建 TypeScript 代码

	const ts_chain = ChatPromptTemplate.fromTemplate(
		`
            请根据以下内容生成 typescript 代码: {base_question}
             ## 注意事项
            1. 生成的代码必须是可执行的
            2. 生成的代码必须按照 javascript 的规范进行编写
            3. 注释必须按照 javascript 的规范进行编写
            `
	)
		.pipe(model.bind({ configurable: { model: 'gpt-4o-mini' } }))
		.pipe(new StringOutputParser())

	const final = ChatPromptTemplate.fromMessages([
		{
			role: 'ai',
			content: '{origin_response}'
		},
		{
			role: 'human',
			content: `
                Java代码: \n {Java} \n
                Python代码: \n {Python} \n
                Typescript代码: \n {Typescript} \n
            `
		},
		{
			role: 'system',
			content: `对生成的代码分别打上评分，并根据语言的优劣性解读其优缺点
               注意：
                 1. 评分输出内容中，需使用源码: 《对应的代码》 来表示对应的代码
            `
		}
	])
		.pipe(model.bind({ configurable: { model: 'gpt-4o' } }))
		.pipe(new StringOutputParser())

	const chain = planner
		.pipe(() =>
			RunnableMap.from({
				Java: java_chain,
				Python: python_chain,
				Typescript: ts_chain
			})
		)
		.pipe(() =>
			RunnablePassthrough.assign({
				origin_response: (input) => input.base_question
			})
		)
		.pipe(final)

	const rs = await chain.invoke({ input: '帮我创建一个随机值' })

	console.log(rs, '--------')
})()
